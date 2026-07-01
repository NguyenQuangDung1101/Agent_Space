from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from database.neo4j_repository import Neo4jSettings, json_text
from neo4j import GraphDatabase
from share.embedder import OllamaEmbedder
from share.schemas import ModificationPlan, ModificationResult


KNOWLEDGE_DOMAIN = "knowledge"
KNOWLEDGE_ROOT_ID = "external_knowledge"
NODE_LABELS = {
    "KnowledgeDocument",
    "KnowledgeChunk",
    "KnowledgeEntity",
}
RELATIONSHIP_TYPES = {
    "HAS_DOCUMENT",
    "HAS_CHUNK",
    "HAS_ENTITY",
    "MENTIONS",
    "RELATED_TO",
    "SUPPORTED_BY",
}
RELATIONSHIP_RULES = {
    "HAS_DOCUMENT": ("Knowledge", "KnowledgeDocument"),
    "HAS_CHUNK": ("KnowledgeDocument", "KnowledgeChunk"),
    "HAS_ENTITY": ("Knowledge", "KnowledgeEntity"),
    "MENTIONS": ("KnowledgeChunk", "KnowledgeEntity"),
    "RELATED_TO": ("KnowledgeEntity", "KnowledgeEntity"),
    "SUPPORTED_BY": ("KnowledgeEntity", "KnowledgeChunk"),
}
TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text)}


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def _without_embedding(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_embedding(item)
            for key, item in value.items()
            if str(key).strip().lower() != "embedding"
        }
    if isinstance(value, list):
        return [_without_embedding(item) for item in value]
    return value


def _decode_properties(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    if isinstance(value, dict):
        return _without_embedding(value)
    try:
        parsed = json.loads(value)
        return _without_embedding(parsed) if isinstance(parsed, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def _node_text(
    node_id: str,
    name: str,
    description: str,
    properties: dict[str, Any],
) -> str:
    return " ".join(
        [node_id, name, description, json_text(_without_embedding(properties))]
    )


class KnowledgeRepository:
    def __init__(
        self,
        settings: Neo4jSettings | None = None,
        embedder: OllamaEmbedder | None = None,
    ) -> None:
        self.settings = settings or Neo4jSettings.from_env()
        self.embedder = embedder or OllamaEmbedder()
        if not self.settings.enabled:
            raise RuntimeError("Neo4j is disabled.")
        self.driver = GraphDatabase.driver(
            self.settings.uri,
            auth=(self.settings.user, self.settings.password),
        )
        self.ensure_schema()

    def close(self) -> None:
        self.driver.close()

    def _run(self, query: str, **parameters: Any) -> list[dict[str, Any]]:
        records, _, _ = self.driver.execute_query(
            query,
            parameters_=parameters,
            database_=self.settings.database,
        )
        return [record.data() for record in records]

    def ensure_schema(self) -> None:
        self.driver.verify_connectivity()
        path = Path(__file__).with_name("knowledge_schema.cypher")
        for statement in path.read_text(encoding="utf-8").split(";"):
            if statement.strip():
                self._run(statement.strip())
        self._run(
            "MERGE (k:Knowledge {id: $id, domain: $domain}) "
            "SET k.name = 'External Knowledge', "
            "k.description = 'Root of the external knowledge graph'",
            id=KNOWLEDGE_ROOT_ID,
            domain=KNOWLEDGE_DOMAIN,
        )
        self._repair_embeddings()

    def _repair_embeddings(self) -> None:
        rows = self._run(
            "MATCH (n {domain: $domain}) "
            "WHERE n.embedding IS NULL OR size(n.embedding) = 64 "
            "RETURN n.id AS id, n.name AS name, n.description AS description, "
            "n.properties_json AS properties",
            domain=KNOWLEDGE_DOMAIN,
        )
        if not rows:
            return
        texts = [
            _node_text(
                row["id"],
                row.get("name") or "",
                row.get("description") or "",
                _decode_properties(row.get("properties")),
            )
            for row in rows
        ]
        for row, embedding in zip(rows, self.embedder.embed_many(texts)):
            self._run(
                "MATCH (n {id: $id, domain: $domain}) "
                "SET n.embedding = $embedding",
                id=row["id"],
                domain=KNOWLEDGE_DOMAIN,
                embedding=embedding,
            )

    @staticmethod
    def _label(labels: list[str]) -> str:
        for label in labels:
            if label in NODE_LABELS or label == "Knowledge":
                return label
        return "KnowledgeEntity"

    def _node(self, node_id: str) -> dict[str, Any] | None:
        rows = self._run(
            "MATCH (n {id: $id, domain: $domain}) "
            "RETURN n.id AS id, labels(n) AS labels, n.name AS name, "
            "n.description AS description, n.properties_json AS properties",
            id=node_id,
            domain=KNOWLEDGE_DOMAIN,
        )
        if not rows:
            return None
        row = rows[0]
        return {
            "id": row["id"],
            "kind": self._label(row["labels"]),
            "name": row.get("name") or "",
            "description": row.get("description") or "",
            "properties": _decode_properties(row.get("properties")),
        }

    def _connected(self, node_id: str) -> bool:
        if node_id == KNOWLEDGE_ROOT_ID:
            return True
        rows = self._run(
            "MATCH (k:Knowledge {id: $root, domain: $domain}), "
            "(n {id: $id, domain: $domain}) "
            "MATCH p = shortestPath((k)-[*..10]-(n)) "
            "WHERE all(r IN relationships(p) WHERE type(r) IN $types) "
            "RETURN count(p) > 0 AS connected",
            root=KNOWLEDGE_ROOT_ID,
            id=node_id,
            domain=KNOWLEDGE_DOMAIN,
            types=sorted(RELATIONSHIP_TYPES),
        )
        return bool(rows and rows[0]["connected"])

    def create_node(
        self,
        node_id: str,
        kind: str,
        name: str,
        description: str = "",
        properties: dict[str, Any] | None = None,
        parent_id: str | None = None,
    ) -> dict[str, Any]:
        if kind not in NODE_LABELS:
            raise ValueError(f"Unsupported knowledge node kind: {kind}")
        if node_id == KNOWLEDGE_ROOT_ID:
            raise ValueError("The Knowledge root cannot be replaced.")
        existing = self._node(node_id)
        if existing and existing["kind"] != kind:
            raise ValueError(
                f"Node ID already exists as {existing['kind']}: {node_id}"
            )
        if kind == "KnowledgeChunk":
            if not parent_id:
                raise ValueError("KnowledgeChunk requires parent_id.")
            parent = self._node(parent_id)
            if not parent or parent["kind"] != "KnowledgeDocument":
                raise ValueError("KnowledgeChunk parent must be a document.")
            if not self._connected(parent_id):
                raise ValueError("Chunk parent is outside the Knowledge graph.")
        properties = _without_embedding(properties or {})
        embedding = self.embedder.embed(
            _node_text(node_id, name, description, properties)
        )
        self._run(
            f"MERGE (n:{kind} {{id: $id, domain: $domain}}) "
            "SET n.name = $name, n.description = $description, "
            "n.properties_json = $properties, n.embedding = $embedding",
            id=node_id,
            domain=KNOWLEDGE_DOMAIN,
            name=name,
            description=description,
            properties=json_text(properties),
            embedding=embedding,
        )
        if kind == "KnowledgeDocument":
            source_id, relationship = KNOWLEDGE_ROOT_ID, "HAS_DOCUMENT"
        elif kind == "KnowledgeEntity":
            source_id, relationship = KNOWLEDGE_ROOT_ID, "HAS_ENTITY"
        else:
            source_id, relationship = parent_id, "HAS_CHUNK"
        self.create_relationship(source_id, node_id, relationship)
        return self._node(node_id) or {"id": node_id, "kind": kind}

    def create_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        properties: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if relationship_type not in RELATIONSHIP_TYPES:
            raise ValueError("Unsupported knowledge relationship type.")
        source = self._node(source_id)
        target = self._node(target_id)
        if not source or not target:
            raise ValueError("Source and target nodes must exist.")
        if source_id != KNOWLEDGE_ROOT_ID and not self._connected(source_id):
            raise ValueError("Source node is outside the Knowledge graph.")
        structural = {"HAS_DOCUMENT", "HAS_ENTITY", "HAS_CHUNK"}
        if relationship_type not in structural and not self._connected(target_id):
            raise ValueError("Target node is outside the Knowledge graph.")
        expected = RELATIONSHIP_RULES[relationship_type]
        if (source["kind"], target["kind"]) != expected:
            raise ValueError(
                f"{relationship_type} requires {expected[0]} -> {expected[1]}."
            )
        self._run(
            f"MATCH (a {{id: $source, domain: $domain}}), "
            f"(b {{id: $target, domain: $domain}}) "
            f"MERGE (a)-[r:{relationship_type}]->(b) "
            "SET r.properties_json = $properties",
            source=source_id,
            target=target_id,
            domain=KNOWLEDGE_DOMAIN,
            properties=json_text(properties or {}),
        )
        return {
            "source_id": source_id,
            "target_id": target_id,
            "type": relationship_type,
        }

    def delete_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
    ) -> dict[str, Any]:
        if relationship_type not in RELATIONSHIP_TYPES:
            raise ValueError("Unsupported knowledge relationship type.")
        if relationship_type in {"HAS_DOCUMENT", "HAS_ENTITY", "HAS_CHUNK"}:
            raise PermissionError(
                "Structural Knowledge relationships cannot be deleted directly; "
                "delete the child node instead."
            )
        if not self._connected(source_id) or not self._connected(target_id):
            raise ValueError("Both nodes must belong to the Knowledge graph.")
        rows = self._run(
            f"MATCH (a {{id: $source, domain: $domain}})"
            f"-[r:{relationship_type}]->"
            f"(b {{id: $target, domain: $domain}}) "
            "WITH collect(r) AS relationships "
            "FOREACH (r IN relationships | DELETE r) "
            "RETURN size(relationships) AS deleted",
            source=source_id,
            target=target_id,
            domain=KNOWLEDGE_DOMAIN,
        )
        return {"deleted": rows[0]["deleted"] if rows else 0}

    def delete_node(self, node_id: str) -> dict[str, Any]:
        if node_id == KNOWLEDGE_ROOT_ID:
            raise PermissionError("The Knowledge root cannot be deleted.")
        node = self._node(node_id)
        if not node or not self._connected(node_id):
            raise ValueError("Node is outside the Knowledge graph.")
        deleted = [node_id]
        if node["kind"] == "KnowledgeDocument":
            rows = self._run(
                "MATCH (:KnowledgeDocument {id: $id, domain: $domain})"
                "-[:HAS_CHUNK]->(chunk:KnowledgeChunk {domain: $domain}) "
                "RETURN collect(chunk.id) AS chunk_ids",
                id=node_id,
                domain=KNOWLEDGE_DOMAIN,
            )
            chunk_ids = rows[0]["chunk_ids"] if rows else []
            if chunk_ids:
                self._run(
                    "MATCH (chunk:KnowledgeChunk {domain: $domain}) "
                    "WHERE chunk.id IN $ids DETACH DELETE chunk",
                    domain=KNOWLEDGE_DOMAIN,
                    ids=chunk_ids,
                )
                deleted.extend(chunk_ids)
        rows = self._run(
            "MATCH (n {id: $id, domain: $domain}) "
            "WITH n, n.id AS deleted_id DETACH DELETE n "
            "RETURN deleted_id",
            id=node_id,
            domain=KNOWLEDGE_DOMAIN,
        )
        return {"deleted": bool(rows), "node_ids": deleted if rows else []}

    def _all_nodes(self) -> list[dict[str, Any]]:
        rows = self._run(
            "MATCH p=(k:Knowledge {id: $root, domain: $domain})-[*0..10]-(n) "
            "WHERE all(r IN relationships(p) WHERE type(r) IN $types) "
            "RETURN DISTINCT n.id AS id, labels(n) AS labels, n.name AS name, "
            "n.description AS description, n.properties_json AS properties, "
            "n.embedding AS embedding LIMIT 2000",
            root=KNOWLEDGE_ROOT_ID,
            domain=KNOWLEDGE_DOMAIN,
            types=sorted(RELATIONSHIP_TYPES),
        )
        return [
            {
                "id": row["id"],
                "kind": self._label(row["labels"]),
                "name": row.get("name") or "",
                "description": row.get("description") or "",
                "properties": _decode_properties(row.get("properties")),
                "embedding": list(row.get("embedding") or []),
            }
            for row in rows
        ]

    def _relationships(self, node_ids: list[str]) -> list[dict[str, Any]]:
        if not node_ids:
            return []
        rows = self._run(
            "MATCH (a {domain: $domain})-[r]-(b {domain: $domain}) "
            "WHERE a.id IN $ids AND b.id IN $ids AND type(r) IN $types "
            "RETURN DISTINCT startNode(r).id AS source_id, "
            "endNode(r).id AS target_id, type(r) AS type, "
            "r.properties_json AS properties",
            domain=KNOWLEDGE_DOMAIN,
            ids=node_ids,
            types=sorted(RELATIONSHIP_TYPES),
        )
        return [
            {
                "source_id": row["source_id"],
                "target_id": row["target_id"],
                "type": row["type"],
                "properties": _decode_properties(row.get("properties")),
            }
            for row in rows
        ]

    def get_anchor_node(self, query: str, top_k: int = 5) -> dict[str, Any]:
        query_tokens = _tokens(query)
        query_embedding = self.embedder.embed(query)
        scored = []
        for node in self._all_nodes():
            if node["id"] == KNOWLEDGE_ROOT_ID:
                continue
            text = _node_text(
                node["id"],
                node["name"],
                node["description"],
                node["properties"],
            )
            node_tokens = _tokens(text)
            lexical = (
                len(query_tokens & node_tokens) / max(1, len(query_tokens))
            )
            semantic = max(0.0, _cosine(query_embedding, node["embedding"]))
            item = {key: value for key, value in node.items() if key != "embedding"}
            item["score"] = round(0.65 * lexical + 0.35 * semantic, 6)
            scored.append(item)
        nodes = sorted(scored, key=lambda item: item["score"], reverse=True)[
            : max(1, min(top_k, 20))
        ]
        ids = [node["id"] for node in nodes]
        return {"nodes": nodes, "relationships": self._relationships(ids)}

    def k_hop_expansion(
        self,
        node_ids: list[str],
        hops: int = 1,
        top_k: int = 30,
    ) -> dict[str, Any]:
        hops = max(1, min(int(hops), 5))
        top_k = max(1, min(int(top_k), 100))
        valid_ids = [node_id for node_id in node_ids if self._connected(node_id)]
        if not valid_ids:
            return {"nodes": [], "relationships": []}
        rows = self._run(
            f"MATCH (s {{domain: $domain}}) WHERE s.id IN $ids "
            f"MATCH p=(s)-[*0..{hops}]-(n {{domain: $domain}}) "
            "WHERE all(r IN relationships(p) WHERE type(r) IN $types) "
            "WITH n, min(length(p)) AS hop "
            "RETURN n.id AS id, labels(n) AS labels, n.name AS name, "
            "n.description AS description, n.properties_json AS properties, "
            "hop ORDER BY hop ASC LIMIT $limit",
            domain=KNOWLEDGE_DOMAIN,
            ids=valid_ids,
            types=sorted(RELATIONSHIP_TYPES),
            limit=top_k,
        )
        nodes = [
            {
                "id": row["id"],
                "kind": self._label(row["labels"]),
                "name": row.get("name") or "",
                "description": row.get("description") or "",
                "properties": _decode_properties(row.get("properties")),
                "hop": row.get("hop", 0),
            }
            for row in rows
        ]
        return {
            "nodes": nodes,
            "relationships": self._relationships([node["id"] for node in nodes]),
        }

    def path_finding(
        self,
        source_id: str,
        target_id: str,
        top_k: int = 3,
        max_hops: int = 8,
    ) -> dict[str, Any]:
        if not self._connected(source_id) or not self._connected(target_id):
            return {"paths": []}
        max_hops = max(1, min(int(max_hops), 10))
        top_k = max(1, min(int(top_k), 10))
        rows = self._run(
            f"MATCH (a {{id: $source, domain: $domain}}), "
            f"(b {{id: $target, domain: $domain}}) "
            f"MATCH p=(a)-[*1..{max_hops}]-(b) "
            "WHERE all(r IN relationships(p) WHERE type(r) IN $types) "
            "RETURN [n IN nodes(p) | {id:n.id, name:n.name, labels:labels(n)}] "
            "AS nodes, [r IN relationships(p) | {type:type(r), "
            "source:startNode(r).id, target:endNode(r).id}] AS relationships, "
            "length(p) AS length ORDER BY length ASC LIMIT $limit",
            source=source_id,
            target=target_id,
            domain=KNOWLEDGE_DOMAIN,
            types=sorted(RELATIONSHIP_TYPES),
            limit=top_k,
        )
        return {"paths": rows}

    def summary(self, limit: int = 50) -> dict[str, Any]:
        nodes = self._all_nodes()[: max(1, min(limit, 200))]
        public_nodes = [
            {key: value for key, value in node.items() if key != "embedding"}
            for node in nodes
        ]
        return {
            "root_id": KNOWLEDGE_ROOT_ID,
            "nodes": public_nodes,
            "relationships": self._relationships(
                [node["id"] for node in public_nodes]
            ),
        }

    def validate(self) -> dict[str, Any]:
        rows = self._run(
            "MATCH (n {domain: $domain}) "
            "WHERE n.id <> $root AND NOT EXISTS { "
            "MATCH (k:Knowledge {id: $root, domain: $domain}) "
            "MATCH p=(k)-[*..10]-(n) "
            "WHERE all(r IN relationships(p) WHERE type(r) IN $types) } "
            "RETURN collect(n.id) AS orphan_ids",
            domain=KNOWLEDGE_DOMAIN,
            root=KNOWLEDGE_ROOT_ID,
            types=sorted(RELATIONSHIP_TYPES),
        )
        orphan_ids = rows[0]["orphan_ids"] if rows else []
        return {
            "valid": not orphan_ids,
            "root_id": KNOWLEDGE_ROOT_ID,
            "orphan_ids": orphan_ids,
        }

    def apply_plan(self, plan: ModificationPlan) -> ModificationResult:
        applied_nodes: list[str] = []
        applied_relationships: list[str] = []
        failures: list[str] = []
        if plan.operation == "import" and plan.import_plan:
            order = {
                "KnowledgeDocument": 0,
                "KnowledgeEntity": 1,
                "KnowledgeChunk": 2,
            }
            nodes = sorted(
                plan.import_plan.nodes,
                key=lambda item: order[item.kind],
            )
            for node in nodes:
                try:
                    self.create_node(
                        node.id,
                        node.kind,
                        node.name,
                        node.description,
                        node.properties,
                        node.parent_id,
                    )
                    applied_nodes.append(node.id)
                except Exception as error:
                    failures.append(f"node {node.id}: {type(error).__name__}: {error}")
            for relationship in plan.import_plan.relationships:
                try:
                    self.create_relationship(
                        relationship.source_id,
                        relationship.target_id,
                        relationship.type,
                        relationship.properties,
                    )
                    applied_relationships.append(
                        f"{relationship.source_id}-[{relationship.type}]->"
                        f"{relationship.target_id}"
                    )
                except Exception as error:
                    failures.append(
                        "relationship "
                        f"{relationship.source_id}-[{relationship.type}]->"
                        f"{relationship.target_id}: {type(error).__name__}: {error}"
                    )
        elif plan.delete_plan:
            deleted_ids = set(plan.delete_plan.node_ids)
            for node_id in plan.delete_plan.node_ids:
                try:
                    self.delete_node(node_id)
                    applied_nodes.append(f"deleted {node_id}")
                except Exception as error:
                    failures.append(f"node {node_id}: {type(error).__name__}: {error}")
            for relationship in plan.delete_plan.relationships:
                if (
                    relationship.source_id in deleted_ids
                    or relationship.target_id in deleted_ids
                ):
                    continue
                try:
                    self.delete_relationship(
                        relationship.source_id,
                        relationship.target_id,
                        relationship.type,
                    )
                    applied_relationships.append(
                        f"deleted {relationship.source_id}-"
                        f"[{relationship.type}]->{relationship.target_id}"
                    )
                except Exception as error:
                    failures.append(
                        f"relationship delete: {type(error).__name__}: {error}"
                    )
        validation = self.validate()
        return ModificationResult(
            success=not failures and validation["valid"],
            applied_nodes=applied_nodes,
            applied_relationships=applied_relationships,
            failures=failures,
            validation=validation,
        )


@lru_cache(maxsize=1)
def get_knowledge_repository() -> KnowledgeRepository:
    return KnowledgeRepository()


def close_knowledge_repository() -> None:
    if get_knowledge_repository.cache_info().currsize:
        get_knowledge_repository().close()
        get_knowledge_repository.cache_clear()
