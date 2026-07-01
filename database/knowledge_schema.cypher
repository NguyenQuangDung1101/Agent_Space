CREATE CONSTRAINT knowledge_root_id IF NOT EXISTS FOR (n:Knowledge) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT knowledge_document_id IF NOT EXISTS FOR (n:KnowledgeDocument) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT knowledge_chunk_id IF NOT EXISTS FOR (n:KnowledgeChunk) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT knowledge_entity_id IF NOT EXISTS FOR (n:KnowledgeEntity) REQUIRE n.id IS UNIQUE;
CREATE INDEX knowledge_entity_name IF NOT EXISTS FOR (n:KnowledgeEntity) ON (n.name);
