from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from share.schemas import (
    Conversation,
    ConversationMemory,
    Message,
    SessionRecord,
    UserContactRequest,
    utc_now,
)


class ConversationStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _json_data(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        if isinstance(value, list):
            return [ConversationStore._json_data(item) for item in value]
        if isinstance(value, dict):
            return {
                key: ConversationStore._json_data(item)
                for key, item in value.items()
            }
        return value

    @classmethod
    def _write(cls, path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                cls._json_data(value),
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _read(path: Path, default: Any = None) -> Any:
        if not path.is_file():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    def conversation_dir(self, conversation_id: str) -> Path:
        return self.root / conversation_id

    def session_dir(self, conversation_id: str, session_id: str) -> Path:
        return self.conversation_dir(conversation_id) / "sessions" / session_id

    def create_conversation(self, title: Optional[str] = None) -> Conversation:
        conversation = Conversation(
            conversation_id=f"conversation_{uuid4().hex[:12]}",
            title=title,
        )
        folder = self.conversation_dir(conversation.conversation_id)
        folder.mkdir(parents=True, exist_ok=False)
        self._write(folder / "conversation.json", conversation)
        self._write(folder / "messages.json", [])
        self._write(
            folder / "memory.json",
            ConversationMemory(conversation_id=conversation.conversation_id),
        )
        return conversation

    def get_conversation(self, conversation_id: str) -> Conversation:
        path = self.conversation_dir(conversation_id) / "conversation.json"
        if not path.is_file():
            raise FileNotFoundError(f"Conversation not found: {conversation_id}")
        return Conversation.model_validate(self._read(path))

    def save_conversation(self, conversation: Conversation) -> None:
        conversation.updated_at = utc_now()
        self._write(
            self.conversation_dir(conversation.conversation_id)
            / "conversation.json",
            conversation,
        )

    def list_conversations(self) -> list[Conversation]:
        conversations = []
        for path in self.root.glob("*/conversation.json"):
            try:
                conversations.append(
                    Conversation.model_validate(self._read(path))
                )
            except Exception:
                continue
        return sorted(
            conversations,
            key=lambda item: item.updated_at,
            reverse=True,
        )

    def add_message(self, message: Message) -> Message:
        conversation = self.get_conversation(message.conversation_id)
        path = self.conversation_dir(message.conversation_id) / "messages.json"
        items = self._read(path, [])
        items.append(message.model_dump(mode="json"))
        self._write(path, items)
        self.save_conversation(conversation)
        return message

    def list_messages(self, conversation_id: str) -> list[Message]:
        self.get_conversation(conversation_id)
        path = self.conversation_dir(conversation_id) / "messages.json"
        return [Message.model_validate(item) for item in self._read(path, [])]

    def get_memory(self, conversation_id: str) -> ConversationMemory:
        self.get_conversation(conversation_id)
        path = self.conversation_dir(conversation_id) / "memory.json"
        data = self._read(path)
        if data is None:
            return ConversationMemory(conversation_id=conversation_id)
        return ConversationMemory.model_validate(data)

    def save_memory(self, memory: ConversationMemory) -> None:
        memory.updated_at = utc_now()
        self._write(
            self.conversation_dir(memory.conversation_id) / "memory.json",
            memory,
        )

    def write_session_json(
        self,
        conversation_id: str,
        session_id: str,
        filename: str,
        value: Any,
    ) -> None:
        self.get_conversation(conversation_id)
        self._write(
            self.session_dir(conversation_id, session_id) / filename,
            value,
        )

    def read_session_json(
        self,
        conversation_id: str,
        session_id: str,
        filename: str,
        default: Any = None,
    ) -> Any:
        return self._read(
            self.session_dir(conversation_id, session_id) / filename,
            default,
        )

    def save_session(self, session: SessionRecord) -> None:
        session.updated_at = utc_now()
        self.write_session_json(
            session.conversation_id,
            session.session_id,
            "session.json",
            session,
        )
        conversation = self.get_conversation(session.conversation_id)
        self.save_conversation(conversation)

    def get_session(
        self,
        session_id: str,
        conversation_id: Optional[str] = None,
    ) -> SessionRecord:
        if conversation_id:
            path = self.session_dir(
                conversation_id,
                session_id,
            ) / "session.json"
            if path.is_file():
                return SessionRecord.model_validate(self._read(path))
        else:
            for path in self.root.glob(
                f"*/sessions/{session_id}/session.json"
            ):
                return SessionRecord.model_validate(self._read(path))
        raise FileNotFoundError(f"Session not found: {session_id}")

    def pending_session(self, conversation_id: str) -> Optional[SessionRecord]:
        sessions_root = self.conversation_dir(conversation_id) / "sessions"
        waiting = []
        for path in sessions_root.glob("*/session.json"):
            try:
                session = SessionRecord.model_validate(self._read(path))
            except Exception:
                continue
            if session.status == "WAITING_FOR_USER":
                waiting.append(session)
        if not waiting:
            return None
        return max(waiting, key=lambda item: item.updated_at)

    def add_contact(self, contact: UserContactRequest) -> None:
        path = self.session_dir(
            contact.conversation_id,
            contact.session_id,
        ) / "contacts.json"
        contacts = self._read(path, [])
        contacts.append(contact.model_dump(mode="json"))
        self._write(path, contacts)

    def update_contact(self, contact: UserContactRequest) -> None:
        path = self.session_dir(
            contact.conversation_id,
            contact.session_id,
        ) / "contacts.json"
        contacts = self._read(path, [])
        replaced = False
        for index, item in enumerate(contacts):
            if item.get("contact_id") == contact.contact_id:
                contacts[index] = contact.model_dump(mode="json")
                replaced = True
                break
        if not replaced:
            contacts.append(contact.model_dump(mode="json"))
        self._write(path, contacts)

    def get_contact(
        self,
        conversation_id: str,
        session_id: str,
        contact_id: str,
    ) -> UserContactRequest:
        path = self.session_dir(conversation_id, session_id) / "contacts.json"
        for item in self._read(path, []):
            if item.get("contact_id") == contact_id:
                return UserContactRequest.model_validate(item)
        raise FileNotFoundError(f"User contact not found: {contact_id}")
