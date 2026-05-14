"""
Web Chat Platform Adapter for Hermes Agent.

A plugin-based gateway adapter that runs an aiohttp HTTP server providing:
- JWT-authenticated API for a React SPA frontend
- Session management with SQLite persistence
- Full markdown + LaTeX rendering support
- Streaming server-sent events for real-time responses
- Static file serving for the built SPA

Architecture:
    SPA (React) ──HTTP──→ Webchat Adapter ──MessageEvent──→ Gateway Runner ──→ AIAgent
                              ↑                                      │
                              └───────────── self.send() ←────────────┘

The adapter starts an HTTP server inside the Hermes gateway process.
Each web chat user maps to a unique chat_id (session UUID), creating
fully isolated conversations independent of other platforms (Telegram,
Discord, etc.).

Configuration (env vars):
    WEBCHAT_PORT         - HTTP server port (default: 8081)
    WEBCHAT_USERNAME     - Login username (required)
    WEBCHAT_PASSWORD     - Login password (required, bcrypt-hashed at startup)
    WEBCHAT_JWT_SECRET   - Secret key for JWT signing (required)
    WEBCHAT_SPA_DIR      - Path to built SPA directory (default: ~/chat-hermes/dist)
"""

import asyncio
import base64
import json
import logging
import mimetypes
import os
import random
import re
import sqlite3
import string
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Hermes gateway imports — these live inside the Hermes package and are
# available when the gateway loads the plugin. We import at module level
# since the gateway's import paths are set up at that point.
# ---------------------------------------------------------------------------
from aiohttp import web
import bcrypt
import jwt

from gateway.platforms.base import (
    BasePlatformAdapter,
    SendResult,
    MessageEvent,
    MessageType,
    ProcessingOutcome,
)
from gateway.session import SessionSource, build_session_key
from gateway.config import PlatformConfig, Platform

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_slug() -> str:
    """Generate a short random URL slug (6 alphanumeric chars)."""
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=6))


def _extract_hostname(url: str) -> str:
    """Extract hostname from a URL, falling back to empty string."""
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc or ""
    except Exception:
        return ""

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default port for the webchat HTTP server
_DEFAULT_PORT = 8081

# JWT token expiration in seconds (24 hours)
_JWT_EXPIRY_SECONDS = 86400

# Path to the SQLite database for session persistence.
# Stored inside the Hermes home directory so it survives gateway restarts.
_DB_PATH = os.path.join(
    os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")),
    "webchat_sessions.db",
)

_UPLOAD_ROOT = os.path.join(
    os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")),
    "webchat_uploads",
)


class _WebchatStreamState:
    """Server-side state for one in-flight assistant response."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.events: list[Dict[str, Any]] = []
        self.subscribers: set[asyncio.Queue] = set()
        self.done = asyncio.Event()
        self.persisted = False
        self.final_content = ""
        self.final_message_id = ""
        self.final_interactions: list[Dict[str, Any]] = []
        self.final_thinking = ""
        self.final_blocks: list[Dict[str, Any]] = []
        self.final_sources: list[Dict[str, Any]] = []
        self.pending_media: list[Dict[str, Any]] = []
        self.error: Optional[str] = None

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self.subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self.subscribers.discard(queue)

    async def publish(self, event: Dict[str, Any]) -> None:
        self.events.append(event)
        self._accumulate(event)
        for queue in list(self.subscribers):
            await queue.put(event)
        if event.get("type") == "done":
            self.done.set()

    def _accumulate(self, event: Dict[str, Any]) -> None:
        item_type = event.get("type")
        item_content = event.get("content") or ""
        if event.get("message_id"):
            self.final_message_id = event.get("message_id") or self.final_message_id
        if item_type == "replace":
            self.final_content = item_content
        elif item_type == "response":
            self.final_content += item_content
        elif item_type == "media":
            self.pending_media.append({
                "path": event.get("path", ""),
                "title": event.get("title", ""),
                "caption": event.get("caption", ""),
                "duration": event.get("duration"),
            })
        elif item_type in ("thinking", "reasoning"):
            self.final_thinking += item_content
            if self.final_blocks and self.final_blocks[-1].get("type") == "text":
                self.final_blocks[-1]["content"] += item_content
            else:
                self.final_blocks.append({"type": "text", "content": item_content})
        elif item_type == "interaction" and event.get("interaction"):
            self.final_content = self.final_content or item_content
            interaction = event["interaction"]
            self.final_interactions.append(interaction)
            if interaction.get("kind") == "tool_call":
                self.final_blocks.append({"type": "tool_call", "interaction": interaction})
        elif item_type == "sources_update":
            self.final_sources = event.get("sources") or []

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _init_db() -> None:
    """Initialise the SQLite database for session storage.
    
    Creates the sessions and messages tables if they don't exist.
    """
    conn = sqlite3.connect(_DB_PATH)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                title      TEXT DEFAULT 'New conversation',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                message_count INTEGER DEFAULT 0,
                pinned INTEGER DEFAULT 0,
                archived INTEGER DEFAULT 0,
                slug TEXT NOT NULL DEFAULT ''
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL DEFAULT '',
                attachments TEXT NOT NULL DEFAULT '[]',
                interactions TEXT NOT NULL DEFAULT '[]',
                reactions TEXT NOT NULL DEFAULT '{}',
                thinking TEXT NOT NULL DEFAULT '',
                blocks TEXT NOT NULL DEFAULT '[]',
                created_at REAL NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                    ON DELETE CASCADE
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_session_created "
            "ON messages(session_id, created_at)"
        )
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(messages)").fetchall()
        }
        if "interactions" not in columns:
            conn.execute(
                "ALTER TABLE messages ADD COLUMN interactions TEXT NOT NULL DEFAULT '[]'"
            )
        if "reactions" not in columns:
            conn.execute(
                "ALTER TABLE messages ADD COLUMN reactions TEXT NOT NULL DEFAULT '{}'"
            )
        if "thinking" not in columns:
            conn.execute(
                "ALTER TABLE messages ADD COLUMN thinking TEXT NOT NULL DEFAULT ''"
            )
        if "blocks" not in columns:
            conn.execute(
                "ALTER TABLE messages ADD COLUMN blocks TEXT NOT NULL DEFAULT '[]'"
            )
        if "sources" not in columns:
            conn.execute(
                "ALTER TABLE messages ADD COLUMN sources TEXT NOT NULL DEFAULT '[]'"
            )

        # Add pinned/archived columns to sessions (if migrating from old schema)
        sess_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
        }
        if "pinned" not in sess_columns:
            conn.execute(
                "ALTER TABLE sessions ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0"
            )
        if "archived" not in sess_columns:
            conn.execute(
                "ALTER TABLE sessions ADD COLUMN archived INTEGER NOT NULL DEFAULT 0"
            )
        if "slug" not in sess_columns:
            conn.execute(
                "ALTER TABLE sessions ADD COLUMN slug TEXT NOT NULL DEFAULT ''"
            )

        # Backfill slugs for existing sessions that have empty slugs.
        # Each session gets a unique slug; retry on collision for each row.
        empty_rows = conn.execute(
            "SELECT session_id FROM sessions WHERE slug IS NULL OR slug = ''"
        ).fetchall()
        for (sid,) in empty_rows:
            for _ in range(5):
                new_slug = _generate_slug()
                existing = conn.execute(
                    "SELECT 1 FROM sessions WHERE slug = ? AND session_id != ?",
                    (new_slug, sid),
                ).fetchone()
                if not existing:
                    conn.execute(
                        "UPDATE sessions SET slug = ? WHERE session_id = ?",
                        (new_slug, sid),
                    )
                    break

        # Add hermes_session_id column to track the Hermes session behind each webchat conversation
        if "hermes_session_id" not in sess_columns:
            conn.execute(
                "ALTER TABLE sessions ADD COLUMN hermes_session_id TEXT DEFAULT ''"
            )

        # Create the user_settings table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id       TEXT PRIMARY KEY,
                accent_color  TEXT DEFAULT '',
                bg_color      TEXT DEFAULT '',
                theme         TEXT DEFAULT 'dark'
            )
        """)
        conn.commit()
    finally:
        conn.close()


def _db_add_session(session_id: str, title: str = "New conversation") -> None:
    """Persist a new session, or ensure an existing session has a slug."""
    now = time.time()
    conn = sqlite3.connect(_DB_PATH)
    try:
        # Check if session already exists
        row = conn.execute(
            "SELECT slug FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            # New session — generate a unique slug
            slug = _generate_slug()
            for _ in range(5):
                slug = _generate_slug()
                existing = conn.execute(
                    "SELECT slug FROM sessions WHERE slug = ?", (slug,)
                ).fetchone()
                if not existing:
                    break
            conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id, title, slug, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, title, slug, now, now),
            )
        elif not row[0]:
            # Existing session with empty slug — backfill one
            slug = _generate_slug()
            for _ in range(5):
                slug = _generate_slug()
                existing = conn.execute(
                    "SELECT slug FROM sessions WHERE slug = ? AND session_id != ?",
                    (slug, session_id),
                ).fetchone()
                if not existing:
                    break
            conn.execute(
                "UPDATE sessions SET slug = ?, updated_at = ? WHERE session_id = ?",
                (slug, now, session_id),
            )
        conn.commit()
    finally:
        conn.close()


def _db_update_session(session_id: str, title: Optional[str] = None) -> None:
    """Update a session's timestamp and optionally its title."""
    now = time.time()
    conn = sqlite3.connect(_DB_PATH)
    try:
        if title:
            conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ?, message_count = message_count + 1 WHERE session_id = ?",
                (title, now, session_id),
            )
        else:
            conn.execute(
                "UPDATE sessions SET updated_at = ?, message_count = message_count + 1 WHERE session_id = ?",
                (now, session_id),
            )
        conn.commit()
    finally:
        conn.close()


def _derive_title(text: str, max_len: int = 90) -> str:
    """Derive a compact conversation title from the first user message.

    Truncates at a sentence or word boundary for clean display.
    """
    clean = " ".join((text or "").strip().split())
    if not clean:
        return "File upload"
    if len(clean) <= max_len:
        return clean
    # Try to cut at sentence boundary within the window
    window = clean[:max_len]
    for sep in (". ", "! ", "? ", "; "):
        idx = window.rfind(sep)
        if idx > max_len // 3:
            return window[:idx + 1]
    # Fall back to word boundary
    idx = window.rfind(" ")
    if idx > max_len // 3:
        return window[:idx] + "..."
    return window + "..."


def _lookup_hermes_session_names_batch(chat_session_ids: list) -> Dict[str, str]:
    """Batch-resolve AI-generated session titles from Hermes state.db.

    Resolution chain (single pass for all sessions):
      1. sessions.json  — maps webchat chat_id → Hermes state.db session_id
      2. state.db       — ``title`` column keyed by Hermes session_id

    Returns a dict mapping chat_session_id → title string for every session
    that has a non-empty title in state.db.
    """
    if not chat_session_ids:
        return {}

    hermes_home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
    sessions_dir = os.path.join(hermes_home, "sessions")
    registry_path = os.path.join(sessions_dir, "sessions.json")
    state_db_path = os.path.join(hermes_home, "state.db")

    # Step 1: Build chat_id → hermes_session_id map from sessions.json
    chat_id_to_hermes_id: Dict[str, str] = {}
    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f) or {}
        id_set = set(chat_session_ids)
        for entry in registry.values():
            origin = entry.get("origin") or {}
            cid = origin.get("chat_id", "")
            if entry.get("platform") == "webchat" and cid in id_set:
                hid = entry.get("session_id", "")
                if hid:
                    chat_id_to_hermes_id[cid] = hid
    except Exception:
        pass

    if not chat_id_to_hermes_id:
        return {}

    # Step 2: Batch query state.db for all Hermes session titles
    hermes_id_to_title: Dict[str, str] = {}
    try:
        conn = sqlite3.connect(f"file:{state_db_path}?mode=ro", uri=True, timeout=2)
        try:
            hermes_ids = list(chat_id_to_hermes_id.values())
            placeholders = ",".join("?" * len(hermes_ids))
            rows = conn.execute(
                f"SELECT id, title FROM sessions WHERE id IN ({placeholders})",
                hermes_ids,
            ).fetchall()
            for row in rows:
                if row[1] and row[1].strip():
                    hermes_id_to_title[row[0]] = row[1].strip()
        finally:
            conn.close()
    except Exception:
        pass

    # Step 3: Combine — map chat_id → title
    return {
        cid: hermes_id_to_title[hid]
        for cid, hid in chat_id_to_hermes_id.items()
        if hid in hermes_id_to_title
    }


def _lookup_hermes_session_name(chat_session_id: str) -> Optional[str]:
    """Single-session convenience wrapper around _lookup_hermes_session_names_batch."""
    result = _lookup_hermes_session_names_batch([chat_session_id])
    return result.get(chat_session_id)


def _db_add_message(
    session_id: str,
    role: str,
    content: str,
    attachments: Optional[list[Dict[str, Any]]] = None,
    interactions: Optional[list[Dict[str, Any]]] = None,
    reactions: Optional[Dict[str, Any]] = None,
    thinking: Optional[str] = None,
    blocks: Optional[list] = None,
    sources: Optional[list] = None,
    message_id: Optional[str] = None,
) -> str:
    """Persist one chat message and update session metadata."""
    now = time.time()
    msg_id = message_id or str(uuid.uuid4())
    encoded_attachments = json.dumps(attachments or [])
    encoded_interactions = json.dumps(interactions or [])
    encoded_reactions = json.dumps(reactions or {})
    encoded_thinking = thinking or ""
    encoded_blocks = json.dumps(blocks or [])
    encoded_sources = json.dumps(sources or [])
    conn = sqlite3.connect(_DB_PATH)
    try:
        if role == "user":
            row = conn.execute(
                "SELECT title FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                slug = _generate_slug()
                conn.execute(
                    "INSERT INTO sessions (session_id, title, slug, created_at, updated_at, message_count) "
                    "VALUES (?, ?, ?, ?, ?, 0)",
                    (session_id, _derive_title(content), slug, now, now),
                )
            elif not row[0] or row[0] == "New conversation":
                conn.execute(
                    "UPDATE sessions SET title = ? WHERE session_id = ?",
                    (_derive_title(content), session_id),
                )
        conn.execute(
            'INSERT OR REPLACE INTO messages '
            '(id, session_id, role, content, attachments, interactions, reactions, thinking, blocks, sources, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (
                msg_id,
                session_id,
                role,
                content or '',
                encoded_attachments,
                encoded_interactions,
                encoded_reactions,
                encoded_thinking,
                encoded_blocks,
                encoded_sources,
                now,
            ),
        )
        conn.execute(
            "UPDATE sessions SET updated_at = ?, message_count = "
            "(SELECT COUNT(*) FROM messages WHERE session_id = ?) "
            "WHERE session_id = ?",
            (now, session_id, session_id),
        )
        conn.commit()
        return msg_id
    finally:
        conn.close()


def _db_get_messages(session_id: str) -> list[Dict[str, Any]]:
    """Return persisted messages for a session."""
    conn = sqlite3.connect(_DB_PATH)
    try:
        rows = conn.execute(
            "SELECT id, role, content, attachments, interactions, "
            "reactions, thinking, blocks, sources, created_at "
            "FROM messages WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,),
        ).fetchall()
        messages = []
        for row in rows:
            try:
                attachments = json.loads(row[3] or "[]")
            except Exception:
                attachments = []
            try:
                interactions = json.loads(row[4] or "[]")
            except Exception:
                interactions = []
            try:
                reactions = json.loads(row[5] or "{}")
            except Exception:
                reactions = {}
            try:
                blocks = json.loads(row[7] or "[]")
            except Exception:
                blocks = []
            try:
                sources_raw = row[8] or "[]"
                sources = json.loads(sources_raw)
            except Exception:
                sources = []
            messages.append({
                "id": row[0],
                "role": row[1],
                "content": row[2],
                "attachments": attachments,
                "interactions": interactions,
                "reactions": reactions,
                "thinking": row[6] or "",
                "blocks": blocks,
                "sources": sources,
                "timestamp": datetime.fromtimestamp(
                    row[9], tz=timezone.utc
                ).isoformat(),
            })
        return messages or _load_hermes_transcript_messages(session_id)
    finally:
        conn.close()


def _load_hermes_transcript_messages(chat_session_id: str) -> list[Dict[str, Any]]:
    """Best-effort import of legacy webchat messages from Hermes transcripts."""
    sessions_dir = os.path.join(
        os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")),
        "sessions",
    )
    registry_path = os.path.join(sessions_dir, "sessions.json")
    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f) or {}
        transcript_id = None
        for entry in registry.values():
            origin = entry.get("origin") or {}
            if (
                entry.get("platform") == "webchat"
                and origin.get("chat_id") == chat_session_id
            ):
                transcript_id = entry.get("session_id")
                break
        if not transcript_id:
            return []
        transcript_path = os.path.join(sessions_dir, f"session_{transcript_id}.json")
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript = json.load(f) or {}
        rows = transcript.get("messages") or []
        messages: list[Dict[str, Any]] = []
        for index, row in enumerate(rows):
            role = row.get("role")
            content = row.get("content") or ""
            if role not in ("user", "assistant") or not content.strip():
                continue
            messages.append({
                "id": f"legacy-{transcript_id}-{index}",
                "role": role,
                "content": content,
                "attachments": [],
                "timestamp": row.get("timestamp")
                    or transcript.get("last_updated")
                    or datetime.now(timezone.utc).isoformat(),
            })
        return messages
    except Exception as e:
        logger.debug("Webchat: legacy transcript load failed for %s: %s", chat_session_id, e)
        return []


def _safe_filename(filename: str) -> str:
    """Return a filesystem-safe upload filename."""
    name = os.path.basename(filename or "upload")
    name = "".join(ch if ch.isalnum() or ch in "._- " else "_" for ch in name)
    return name.strip() or "upload"


def _file_token(session_id: str, filename: str) -> str:
    raw = f"{session_id}/{filename}".encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_file_token(token: str) -> Optional[tuple[str, str]]:
    try:
        padded = token + ("=" * (-len(token) % 4))
        value = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
        session_id, filename = value.split("/", 1)
        return session_id, filename
    except Exception:
        return None


def _build_upload_placeholder(media_paths: list[str], media_types: list[str]) -> str:
    """Build a readable text placeholder for media-only webchat sends."""
    parts: list[str] = []
    for i, path in enumerate(media_paths):
        mtype = media_types[i] if i < len(media_types) else ""
        name = os.path.basename(path)
        if mtype.startswith("image/"):
            parts.append(f"[User uploaded an image: {name}]")
        elif mtype.startswith("audio/"):
            parts.append(f"[User uploaded audio: {name}]")
        else:
            parts.append(f"[User uploaded a file: {name}]")
    return "\n".join(parts)


async def _read_uploads(request: web.Request, session_id: str) -> tuple[str, list[Dict[str, Any]], list[str], list[str]]:
    """Read either JSON or multipart webchat input.
    
    Returns: message_text, attachment metadata for the UI, local media paths,
    and media MIME types for Hermes' MessageEvent.
    """
    content_type = request.content_type or ""
    if content_type.startswith("multipart/"):
        reader = await request.multipart()
        message_text = ""
        attachments: list[Dict[str, Any]] = []
        media_paths: list[str] = []
        media_types: list[str] = []
        upload_dir = os.path.join(_UPLOAD_ROOT, session_id)
        os.makedirs(upload_dir, exist_ok=True)
        
        async for part in reader:
            if part.name == "message":
                message_text = (await part.text()).strip()
                continue
            if part.name == "session_id":
                # The caller already resolved session_id from query/body.
                continue
            if part.name != "files":
                continue
            
            original_name = _safe_filename(part.filename or "upload")
            stored_name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}_{original_name}"
            local_path = os.path.join(upload_dir, stored_name)
            size = 0
            with open(local_path, "wb") as f:
                while True:
                    chunk = await part.read_chunk()
                    if not chunk:
                        break
                    size += len(chunk)
                    f.write(chunk)
            mime_type = part.headers.get("Content-Type") or mimetypes.guess_type(original_name)[0] or "application/octet-stream"
            media_paths.append(local_path)
            media_types.append(mime_type)
            attachments.append({
                "name": original_name,
                "type": mime_type,
                "size": size,
                "url": f"/api/files/{_file_token(session_id, stored_name)}",
            })
        return message_text, attachments, media_paths, media_types
    
    data = await request.json()
    message_text = data.get("message", "").strip()
    return message_text, [], [], []


def _db_get_sessions(filter_by: str = "") -> list[Dict[str, Any]]:
    """Return sessions ordered by most recently updated first.

    Args:
        filter_by: If "archived", return only archived sessions.
                   Otherwise return only non-archived sessions.
    """
    conn = sqlite3.connect(_DB_PATH)
    try:
        where_clause = ""
        if filter_by == "archived":
            where_clause = "WHERE COALESCE(s.archived, 0) = 1"
        else:
            where_clause = "WHERE COALESCE(s.archived, 0) = 0"

        rows = conn.execute(f"""
            SELECT
                s.session_id,
                s.title,
                s.created_at,
                s.updated_at,
                COALESCE(NULLIF(COUNT(m.id), 0), s.message_count),
                s.pinned,
                s.archived,
                s.slug
            FROM sessions s
            LEFT JOIN messages m ON m.session_id = s.session_id
            {where_clause}
            GROUP BY s.session_id
            ORDER BY COALESCE(s.pinned, 0) DESC, s.updated_at DESC
        """).fetchall()
        # Batch-resolve AI-generated session names from Hermes state.db in one pass
        all_session_ids = [row[0] for row in rows]
        hermes_titles = _lookup_hermes_session_names_batch(all_session_ids)

        sessions = []
        to_persist: list = []  # (session_id, title) pairs to write back
        for row in rows:
            session_id = row[0]
            title = row[1]
            count = row[4]

            # Prefer the AI-generated title from Hermes state.db if it differs
            hermes_name = hermes_titles.get(session_id)
            if hermes_name and hermes_name != title:
                title = hermes_name
                to_persist.append((title[:120], session_id))
            elif not title or title == "New conversation":
                # Fall back to first user message from the legacy transcript
                legacy_messages = _load_hermes_transcript_messages(session_id)
                first_user = next(
                    (m.get("content", "") for m in legacy_messages if m.get("role") == "user"),
                    "",
                )
                if first_user:
                    title = _derive_title(first_user)
                    to_persist.append((title[:120], session_id))
                if legacy_messages and not count:
                    count = len(legacy_messages)

            sessions.append({
                "session_id": session_id,
                "title": title,
                "created_at": row[2],
                "updated_at": row[3],
                "message_count": count,
                "pinned": bool(row[5]) if len(row) > 5 else False,
                "archived": bool(row[6]) if len(row) > 6 else False,
                "slug": row[7] if len(row) > 7 else "",
            })

        # Persist discovered/enriched titles back into the webchat DB in one batch
        if to_persist:
            try:
                conn.executemany(
                    "UPDATE sessions SET title = ? WHERE session_id = ?",
                    to_persist,
                )
                conn.commit()
            except Exception:
                pass
        return sessions
    finally:
        conn.close()


def _db_delete_session(session_id: str) -> bool:
    """Delete a session. Returns True if a row was deleted."""
    conn = sqlite3.connect(_DB_PATH)
    try:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        cursor = conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def _db_get_user_settings(user_id: str) -> Dict[str, Any]:
    """Return persisted settings for a user, or defaults if none exist."""
    conn = sqlite3.connect(_DB_PATH)
    try:
        row = conn.execute(
            "SELECT accent_color, bg_color, theme FROM user_settings WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if row:
            return {
                "accent_color": row[0] or "",
                "bg_color": row[1] or "",
                "theme": row[2] or "dark",
            }
        return {"accent_color": "", "bg_color": "", "theme": "dark"}
    finally:
        conn.close()


def _db_set_user_settings(user_id: str, settings: Dict[str, str]) -> Dict[str, Any]:
    """Persist user settings, inserting or updating as needed."""
    conn = sqlite3.connect(_DB_PATH)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO user_settings (user_id, accent_color, bg_color, theme) "
            "VALUES (?, ?, ?, ?)",
            (
                user_id,
                settings.get("accent_color", ""),
                settings.get("bg_color", ""),
                settings.get("theme", "dark"),
            ),
        )
        conn.commit()
        row = conn.execute(
            "SELECT accent_color, bg_color, theme FROM user_settings WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        return {
            "accent_color": row[0] or "",
            "bg_color": row[1] or "",
            "theme": row[2] or "dark",
        }
    finally:
        conn.close()


def _db_rename_session(session_id: str, title: str) -> bool:
    """Rename a session. Returns True if a row was updated."""
    conn = sqlite3.connect(_DB_PATH)
    try:
        cursor = conn.execute(
            "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?",
            (title.strip()[:60] or "New conversation", time.time(), session_id),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def _db_toggle_pin(session_id: str) -> bool:
    """Toggle the pinned status of a session. Returns True if a row was updated."""
    conn = sqlite3.connect(_DB_PATH)
    try:
        row = conn.execute(
            "SELECT pinned FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if not row:
            return False
        new_val = 1 if not row[0] else 0
        conn.execute(
            "UPDATE sessions SET pinned = ?, updated_at = ? WHERE session_id = ?",
            (new_val, time.time(), session_id),
        )
        conn.commit()
        return True
    finally:
        conn.close()


def _db_toggle_archive(session_id: str) -> bool:
    """Toggle the archived status of a session. Returns True if a row was updated."""
    conn = sqlite3.connect(_DB_PATH)
    try:
        row = conn.execute(
            "SELECT archived FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if not row:
            return False
        new_val = 1 if not row[0] else 0
        conn.execute(
            "UPDATE sessions SET archived = ?, updated_at = ? WHERE session_id = ?",
            (new_val, time.time(), session_id),
        )
        conn.commit()
        return True
    finally:
        conn.close()


def _db_set_reaction(message_id: str, reaction: str, user_id: str) -> Dict[str, Any]:
    """Toggle a user reaction for one persisted message."""
    conn = sqlite3.connect(_DB_PATH)
    try:
        row = conn.execute(
            "SELECT reactions FROM messages WHERE id = ?",
            (message_id,),
        ).fetchone()
        if not row:
            raise KeyError("Message not found")
        try:
            reactions = json.loads(row[0] or "{}")
        except Exception:
            reactions = {}
        users = set(reactions.get(reaction) or [])
        if user_id in users:
            users.remove(user_id)
        else:
            users.add(user_id)
        if users:
            reactions[reaction] = sorted(users)
        else:
            reactions.pop(reaction, None)
        conn.execute(
            "UPDATE messages SET reactions = ? WHERE id = ?",
            (json.dumps(reactions), message_id),
        )
        conn.commit()
        return reactions
    finally:
        conn.close()


def _db_update_interaction(interaction_id: str, interaction: Dict[str, Any]) -> None:
    """Persist an updated interaction payload inside any message containing it."""
    conn = sqlite3.connect(_DB_PATH)
    try:
        rows = conn.execute(
            "SELECT id, interactions FROM messages WHERE interactions LIKE ?",
            (f"%{interaction_id}%",),
        ).fetchall()
        for message_id, encoded in rows:
            try:
                interactions = json.loads(encoded or "[]")
            except Exception:
                continue
            changed = False
            for index, item in enumerate(interactions):
                if item.get("id") == interaction_id:
                    interactions[index] = interaction
                    changed = True
            if changed:
                conn.execute(
                    "UPDATE messages SET interactions = ? WHERE id = ?",
                    (json.dumps(interactions), message_id),
                )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def _hash_password(password: str) -> str:
    """Hash a password with bcrypt (auto-generates a salt)."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(password: str, password_hash: str) -> bool:
    """Check a password against its bcrypt hash."""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def _create_jwt(username: str, secret: str) -> str:
    """Create a JWT token for an authenticated user.
    
    The token includes the username, a unique user ID, and an expiration
    timestamp. All subsequent API calls use this token for auth.
    """
    payload = {
        "username": username,
        "user_id": f"user-{username}",
        "exp": int(time.time()) + _JWT_EXPIRY_SECONDS,
        "iat": int(time.time()),
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def _verify_jwt(token: str, secret: str) -> Optional[Dict[str, Any]]:
    """Verify a JWT token and return its payload, or None if invalid/expired."""
    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"])
        return payload
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None


# ---------------------------------------------------------------------------
# Webchat Platform Adapter
# ---------------------------------------------------------------------------

class WebchatAdapter(BasePlatformAdapter):
    """Web Chat platform adapter for Hermes Agent.
    
    This adapter runs an aiohttp HTTP server that:
    1. Serves the React SPA as static files
    2. Provides a JWT-authenticated API for sending/receiving messages
    3. Bridges web chat messages to the Hermes gateway via MessageEvent
    4. Pushes agent responses back to the web client via SSE streaming
    
    Sessions are fully isolated from other platforms (Telegram, Discord, etc.)
    because the adapter uses unique chat_ids for each web chat conversation.
    """
    
    MAX_MESSAGE_LENGTH = 50000
    
    # Tell the gateway's stream consumer to keep MEDIA:/path markers inline
    # in the streaming text instead of stripping them and delivering files
    # as separate native attachments.  The webchat frontend's parseMediaMarkers()
    # renders them at their original positions in the text flow.
    KEEP_MEDIA_INLINE = True
    # Tells the stream consumer NOT to strip <think>...</think> blocks from
    # streaming content so the adapter can parse them into separate reasoning
    # SSE events for the webchat frontend's ThinkingDisclosure.
    PRESERVE_REASONING = True
    
    def __init__(self, config: PlatformConfig):
        """Initialise the webchat adapter.
        
        Args:
            config: PlatformConfig from Hermes gateway config.yaml.
                    The 'extra' dict can contain: port, host, jwt_secret,
                    spa_dir, username, password_hash.
        """
        # Initialise the base adapter with our custom platform type
        super().__init__(config, Platform("webchat"))
        
        # Read configuration from env vars (preferred) or config.yaml extra dict
        extra = config.extra or {}
        self._port = int(os.environ.get("WEBCHAT_PORT") or extra.get("port", _DEFAULT_PORT))
        self._host = extra.get("host", "0.0.0.0")
        self._spa_dir = os.environ.get("WEBCHAT_SPA_DIR") or extra.get(
            "spa_dir", os.path.expanduser("~/chat-hermes/dist")
        )
        
        # Auth configuration
        self._username = os.environ.get("WEBCHAT_USERNAME") or extra.get("username", "")
        password_plain = os.environ.get("WEBCHAT_PASSWORD") or ""
        self._password_hash = extra.get("password_hash") or (
            _hash_password(password_plain) if password_plain else ""
        )
        self._jwt_secret = os.environ.get("WEBCHAT_JWT_SECRET") or extra.get("jwt_secret", "")
        
        # aiohttp server state
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        
        # In-flight streams live independently of any browser tab.  SSE
        # handlers subscribe to these states; the gateway task continues to
        # publish and persist even if every subscriber disconnects.
        self._active_streams: Dict[str, _WebchatStreamState] = {}
        self._pending_interactions: Dict[str, Dict[str, Any]] = {}
        # Blocking clarify/plan events — keyed by interaction_id
        self._pending_clarify: Dict[str, threading.Event] = {}
        self._pending_clarify_values: Dict[str, str] = {}
        # Search sources state — keyed by chat_id (session_id)
        self._sources_state: Dict[str, list[Dict[str, str]]] = {}
        # Main event loop — set during connect()
        self._loop: asyncio.AbstractEventLoop | None = None
    
    # ── Lifecycle ───────────────────────────────────────────────────────
    
    async def connect(self) -> bool:
        """Start the aiohttp HTTP server.
        
        Called by the gateway runner when the platform is enabled.
        Sets up all routes and begins listening.
        """
        try:
            _init_db()
        except Exception as e:
            logger.error("Webchat: Failed to initialise database: %s", e)
            return False
        
        # Store reference to the asyncio event loop for cross-thread callbacks
        self._loop = asyncio.get_running_loop()
        # Validate configuration
        if not self._username or not self._password_hash or not self._jwt_secret:
            logger.error(
                "Webchat: Missing configuration. "
                "Set WEBCHAT_USERNAME, WEBCHAT_PASSWORD, and WEBCHAT_JWT_SECRET env vars."
            )
            return False
        
        # Build the aiohttp application with routes
        self._app = web.Application()
        self._setup_routes()
        
        # Start the HTTP server
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()
        
        logger.info("Webchat: HTTP server listening on %s:%s", self._host, self._port)
        logger.info("Webchat: SPA directory: %s", self._spa_dir)
        self._mark_connected()
        return True
    
    async def disconnect(self) -> None:
        """Shut down the HTTP server cleanly."""
        if self._runner:
            await self._runner.cleanup()
        self._mark_disconnected()
        logger.info("Webchat: HTTP server stopped")
    
    # ── Message handling ─────────────────────────────────────────────────
    
    _THINK_RE = re.compile(
        r'<(REASONING_SCRATCHPAD|think|reasoning|THINKING|thinking|thought)>'
        r'(.*?)</\1>',
        re.DOTALL,
    )
    
    def _extract_reasoning(self, content: str) -> tuple[str, str]:
        """Extract <think>...</think> reasoning blocks from content.
        
        Returns (display_text, thinking_text) where display_text has think
        blocks stripped and thinking_text is the concatenated reasoning.
        """
        thinking_parts = []
        def _replace_think(match):
            thinking_parts.append(match.group(2).strip())
            return ""
        display = self._THINK_RE.sub(_replace_think, content).strip()
        display = re.sub(r'\n{3,}', '\n\n', display).strip()
        return display, "\n".join(thinking_parts)
    
    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Called by the gateway runner when the agent has a response.
        
        This is the callback path — when the AIAgent finishes processing
        a message, the gateway calls adapter.send() to deliver the response.
        We push the response into the pending queue for the associated
        HTTP request to pick up and stream to the client.
        
        Args:
            chat_id: The session_id that was used in the original MessageEvent.
            content: The agent's response text.
            reply_to: Optional message_id this is replying to.
            metadata: Optional extra data from the gateway.
        
        Returns:
            SendResult indicating success.
        """
        # MEDIA:/path markers pass through inline so the frontend's
        # parseMediaMarkers() can render attachments at their original
        # position in the text flow — no stripping needed here.
        state = self._active_streams.get(chat_id)
        if state is None:
            logger.warning("Webchat: No active stream for session %s", chat_id)
            return SendResult(success=False, message_id="")
        
        # Push the response content as an SSE event.  The HTTP handler owns the
        # stream lifecycle now, so adapter.send() must not close the queue; the
        # gateway may call send() followed by edit_message() for token streaming.
        message_id = str(uuid.uuid4())
        # Extract reasoning blocks from send() content too — commentary and
        # first-send text may contain think blocks when PRESERVE_REASONING
        # is enabled.
        display_content, thinking_text = self._extract_reasoning(content)
        if thinking_text:
            try:
                await state.publish({
                    "type": "reasoning",
                    "content": thinking_text,
                    "session_id": chat_id,
                })
            except Exception:
                pass
        try:
            await state.publish({
                "type": "response",
                "content": display_content,
                "session_id": chat_id,
                "message_id": message_id,
            })
        except Exception as e:
            logger.error("Webchat: Failed to push response: %s", e)
            return SendResult(success=False, message_id="")
        
        return SendResult(success=True, message_id=message_id)
    
    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        *,
        finalize: bool = False,
    ) -> SendResult:
        """Stream an edited version of an in-flight webchat message.
        
        Hermes' gateway stream consumer uses platform message editing as its
        streaming transport: first it sends a preview, then it edits the same
        platform message as deltas arrive.  For the web UI this maps naturally
        to a "replace the current streamed assistant message" SSE event.
        """
        # MEDIA:/path markers pass through inline — same rationale as send().
        state = self._active_streams.get(chat_id)
        if state is None:
            logger.warning("Webchat: No active stream for edit in session %s", chat_id)
            return SendResult(success=False, message_id=message_id, error="No pending response")
        
        # Extract <think>...</think> reasoning blocks from content and push
        # them as separate "reasoning" SSE events so the frontend can render
        # them in the ThinkingDisclosure rather than in the main text.
        # The stream consumer retains these blocks when PRESERVE_REASONING is
        # True (set by WebchatAdapter).
        display_content, thinking_text = self._extract_reasoning(content)
        if thinking_text:
            try:
                await state.publish({
                    "type": "reasoning",
                    "content": thinking_text,
                    "session_id": chat_id,
                })
            except Exception:
                pass
        
        try:
            await state.publish({
                "type": "replace",
                "content": display_content,
                "session_id": chat_id,
                "message_id": message_id,
                "final": finalize,
            })
            return SendResult(success=True, message_id=message_id)
        except Exception as e:
            logger.error("Webchat: Failed to push edit: %s", e)
            return SendResult(success=False, message_id=message_id, error=str(e))

    # -----------------------------------------------------------------------
    # Media file delivery — push structured SSE events instead of the base
    # adapter's emoji-prefixed text fallback.  The frontend handles "media"
    # events natively, using /api/media to serve the file.  This avoids
    # leaking emoji markers (🖼️, 📎, 🎬, 🎵) into message.content, which
    # would cause broken attachment rendering when the source text contains
    # MEDIA: references from reasoning or internal tool output. (#media-reasoning-leak)
    # -----------------------------------------------------------------------

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Files are already referenced inline via MEDIA:/path markers in the
        streaming text — no separate "media" SSE event needed.  The gateway
        calls this when it detects MEDIA: tags, but the webchat frontend
        handles them natively through parseMediaMarkers()."""
        return SendResult(success=True, message_id="")

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Same as send_document — inline MEDIA: markers replace the need for
        separate media events on the webchat platform."""
        return SendResult(success=True, message_id="")

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        duration: Optional[float] = None,
        waveform: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Deliver a voice/audio file by pushing a structured 'media' SSE event."""
        state = self._active_streams.get(chat_id)
        if state:
            try:
                await state.publish({
                    "type": "media",
                    "path": audio_path,
                    "title": os.path.basename(audio_path),
                    "duration": duration,
                })
                return SendResult(success=True, message_id="")
            except Exception as e:
                logger.error("Webchat: send_voice failed: %s", e)
        return SendResult(success=False, message_id="")

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Deliver a video file by pushing a structured 'media' SSE event."""
        state = self._active_streams.get(chat_id)
        if state:
            try:
                await state.publish({
                    "type": "media",
                    "path": video_path,
                    "title": os.path.basename(video_path),
                    "caption": caption or "",
                })
                return SendResult(success=True, message_id="")
            except Exception as e:
                logger.error("Webchat: send_video failed: %s", e)
        return SendResult(success=False, message_id="")

    async def send_typing(
        self,
        chat_id: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Send a typing indicator to the web client.
        
        Pushes a 'typing' event to the SSE stream so the UI can show
        "Hermes is thinking..." while the agent processes the request.
        """
        state = self._active_streams.get(chat_id)
        if state:
            try:
                await state.publish({"type": "typing", "session_id": chat_id})
            except Exception:
                pass

    async def send_reasoning(
        self,
        chat_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Stream a reasoning/thinking delta to the web client.

        Called by the GatewayStreamConsumer for each ``delta.reasoning_content``
        chunk.  Pushes a ``reasoning`` SSE event that the frontend accumulates
        into its collapsible "Thinking..." disclosure block.
        """
        state = self._active_streams.get(chat_id)
        if state:
            try:
                await state.publish({
                    "type": "reasoning",
                    "content": content,
                    "session_id": chat_id,
                })
            except Exception:
                pass

    async def send_tool_call(
        self,
        chat_id: str,
        tool: str,
        args: Dict[str, Any],
        result_summary: str,
    ) -> None:
        """Push a tool_call interaction SSE event for a tool invocation.

        Called by the gateway runner when the agent uses a tool so the
        frontend can display a real-time tool usage indicator in the
        reasoning block.
        """
        args_preview = json.dumps(args, ensure_ascii=False, default=str)
        if len(args_preview) > 200:
            args_preview = args_preview[:197] + "..."
        interaction = self._register_interaction(
            chat_id=chat_id,
            kind="tool_call",
            title=f"Tool: {tool}",
            content=args_preview,
            controls=[],
            state={"tool": tool, "args": json.loads(json.dumps(args, default=str))},
        )
        await self._queue_event(chat_id, {
            "type": "interaction",
            "session_id": chat_id,
            "content": f"**{tool}**\n{args_preview}",
            "interaction": interaction,
        })

        # ── Sources: intercept search tool results ───────────────────────
        search_tools = {
            "mcp_brave_search_brave_web_search",
            "mcp_brave_search_brave_local_search",
            "mcp_brave_llm_context_brave_llm_context",
            "brave_web_search", "web_search",
            "mcp_brave_search_brave_news_search",
            "mcp_brave_search_brave_video_search",
        }
        if tool in search_tools:
            sources = self._extract_sources(result_summary, args)
            if sources:
                existing = self._sources_state.get(chat_id, [])
                seen_urls = {s["url"] for s in existing}
                for s in sources:
                    if s["url"] not in seen_urls:
                        existing.append(s)
                        seen_urls.add(s["url"])
                self._sources_state[chat_id] = existing[-5:]
                await self._queue_event(chat_id, {
                    "type": "sources_update",
                    "session_id": chat_id,
                    "sources": self._sources_state[chat_id],
                })

    async def send_tool_result(
        self,
        chat_id: str,
        tool: str,
        function_result: Any,
    ) -> None:
        """Process a completed search tool's actual result and emit sources.

        Called by tool_complete_callback in the gateway runner — fires
        *after* the tool returns, so ``function_result`` contains the real
        search results (unlike ``send_tool_call`` which only has the args).
        """
        search_tools = {
            "mcp_brave_search_brave_web_search",
            "mcp_brave_search_brave_local_search",
            "mcp_brave_llm_context_brave_llm_context",
            "brave_web_search",
            "web_search",
            "mcp_brave_search_brave_news_search",
            "mcp_brave_search_brave_video_search",
        }
        if tool not in search_tools:
            return

        result_str = ""
        if isinstance(function_result, str):
            result_str = function_result
        elif isinstance(function_result, dict):
            result_str = json.dumps(function_result, default=str)
        elif isinstance(function_result, (list, tuple)):
            result_str = json.dumps(function_result, default=str)
        else:
            result_str = str(function_result)

        sources = self._extract_sources(result_str, None)
        if not sources:
            return

        existing = self._sources_state.get(chat_id, [])
        seen_urls = {s["url"] for s in existing}
        for s in sources:
            if s["url"] not in seen_urls:
                existing.append(s)
                seen_urls.add(s["url"])
        self._sources_state[chat_id] = existing[-5:]
        await self._queue_event(chat_id, {
            "type": "sources_update",
            "session_id": chat_id,
            "sources": self._sources_state[chat_id],
        })

    @staticmethod
    def _extract_sources(result_summary: str, args: Dict[str, Any] | None = None) -> list[Dict[str, str]]:
        """
        Parse search tool result summaries into structured source objects.

        Tries two approaches:
        1. If result_summary is JSON with a ``results`` array, extract URLs.
        2. If args has a ``query`` field, construct a Brave search link card.
        """
        import json as _json
        import urllib.parse

        text = (result_summary or "").strip()
        sources: list[Dict[str, str]] = []

        # Helper to extract hostname from URL
        def _hostname(url: str) -> str:
            try:
                return urllib.parse.urlparse(url).netloc
            except Exception:
                return url

        # Approach 1: full search result JSON with a results array
        # Unwrap MCP `{"result": "..."}` envelope: the MCP tool handler
        # (mcp_tool.py:2272) wraps ALL tool output in ``{"result": ...}``,
        # double-encoding inner JSON as a string.  Peel the envelope so we
        # can reach the actual search results.
        inner_text: str | None = None
        if text.startswith("{") or text.startswith("["):
            try:
                data = _json.loads(text)
                # Peel MCP wrapper if present
                if "result" in data and isinstance(data["result"], str):
                    inner_val = data["result"].strip()
                    if inner_val.startswith(("{", "[")):
                        try:
                            inner = _json.loads(inner_val)
                            if isinstance(inner, list):
                                data = {"results": inner}
                            elif isinstance(inner, dict):
                                data = inner
                        except _json.JSONDecodeError:
                            pass
                    else:
                        # Non-JSON MCP envelope (brave_llm_context plain text)
                        inner_text = inner_val
                raw_results = data.get("results", data.get("web", {}).get("results", []))
                if isinstance(raw_results, list):
                    for r in raw_results:
                        url = r.get("url", "")
                        if url:
                            snippet = r.get("description", r.get("snippet", r.get("content", "")))
                            sources.append({
                                "title": r.get("title", url)[:120],
                                "url": url,
                                "snippet": str(snippet)[:200] if snippet else "",
                                "domain": r.get("source", r.get("hostname", _hostname(url))),
                            })
                if sources:
                    return sources
            except json.JSONDecodeError:
                pass

        # Approach 2: brave_llm_context plain-text format
        # The LLM Context MCP returns results as:
        #   ## Result N: Title\nURL: https://...\n\nContent...\n\n---\n
        llm_text = inner_text or text
        if not sources and ("## Result " in llm_text):
            import re as _re
            for match in _re.finditer(
                r'##\s+Result\s+\d+:\s*(.+?)\s*\n\s*URL:\s*(\S+)',
                llm_text,
            ):
                title = match.group(1).strip()[:120]
                url = match.group(2).strip()
                if url:
                    sources.append({
                        "title": title or url[:120],
                        "url": url,
                        "snippet": "",
                        "domain": _hostname(url),
                    })
            if sources:
                return sources

        # Approach 3: tool args with a query field (fallback search link)
        if args and args.get("query"):
            q = args["query"]
            q_encoded = urllib.parse.quote(q[:200])
            sources.append({
                "title": f"Search: {q[:80]}",
                "url": f"https://search.brave.com/search?q={q_encoded}",
                "snippet": "",
                "domain": "Brave Search",
            })
        return sources

    async def send_clarify(
        self,
        chat_id: str,
        question: str,
        choices: list[str] | None = None,
        timeout: float = 120.0,
    ) -> str:
        """
        Present a multi-choice or open-ended question to the user and wait
        for a response. Blocks the agent via threading.Event.

        Returns the user's selected value, or "(clarify timed out)" on timeout.
        """
        controls: list[Dict[str, Any]] = []
        if choices:
            for i, choice in enumerate(choices):
                controls.append({
                    "label": choice,
                    "value": choice,
                    "variant": "secondary",
                })
        else:
            controls.append({
                "label": "Type your answer...",
                "value": "__open__",
                "variant": "ghost",
            })

        interaction = self._register_interaction(
            chat_id=chat_id,
            kind="clarify_response",
            title="🤔 " + question,
            content="",
            controls=controls,
        )
        interaction_id = interaction["id"]
        # Use threading.Event for cross-thread signalling (agent runs in executor)
        import threading as _threading
        event = _threading.Event()
        self._pending_clarify[interaction_id] = event
        self._pending_clarify_values[interaction_id] = ""

        _db_add_message(chat_id, "assistant", f"**{question}**", interactions=[interaction])

        queued = await self._queue_event(chat_id, {
            "type": "interaction",
            "session_id": chat_id,
            "content": question,
            "interaction": interaction,
        })

        # Wait in a thread so the asyncio event loop isn't blocked
        import concurrent.futures as _futures
        loop = asyncio.get_running_loop()
        def _wait():
            return event.wait(timeout=timeout)
        waited = await loop.run_in_executor(None, _wait)

        if not waited:
            self._pending_clarify.pop(interaction_id, None)
            self._pending_clarify_values.pop(interaction_id, None)
            return "(clarify timed out)"

        value = self._pending_clarify_values.pop(interaction_id, "")
        self._pending_clarify.pop(interaction_id, None)
        return value

    def create_clarify_callback(self, chat_id: str):
        """
        Return a callable (question, choices) -> str suitable for passing
        as AIAgent's clarify_callback.

        The callback is called from the agent's executor thread. send_clarify
        uses threading.Event for cross-thread signalling and schedules the
        actual async work on the main event loop.
        """
        adapter = self

        def _cb(question: str, choices: list[str] | None = None) -> str:
            coro = adapter.send_clarify(chat_id, question, choices)
            if adapter._loop is not None:
                future = asyncio.run_coroutine_threadsafe(coro, adapter._loop)
                return future.result()
            return asyncio.run(coro)

        return _cb

    async def get_chat_info(self, chat_id: str) -> Dict[str, str]:
        """Return metadata about a chat conversation."""
        return {"name": f"Web Chat ({chat_id[:8]}...)", "type": "dm"}

    async def _queue_event(self, chat_id: str, event: Dict[str, Any]) -> bool:
        """Push an event into an active webchat stream."""
        state = self._active_streams.get(chat_id)
        if not state:
            return False
        event.setdefault("session_id", chat_id)
        await state.publish(event)
        return True

    async def _stream_state_to_response(
        self,
        request: web.Request,
        state: _WebchatStreamState,
    ) -> web.StreamResponse:
        """Attach one HTTP/SSE response to an in-flight stream state."""
        resp = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "X-Accel-Buffering": "no",
            },
        )
        await resp.prepare(request)

        queue = state.subscribe()
        try:
            for event in list(state.events):
                await resp.write(f"data: {json.dumps(event)}\n\n".encode())
                await resp.drain()

            while not state.done.is_set():
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    await resp.write(b": keep-alive\n\n")
                    await resp.drain()
                    continue

                await resp.write(f"data: {json.dumps(event)}\n\n".encode())
                await resp.drain()

            # Flush events published while replay was being written, including
            # the final done event if it won the race with the live loop.
            while not queue.empty():
                event = queue.get_nowait()
                await resp.write(f"data: {json.dumps(event)}\n\n".encode())
                await resp.drain()
        except (ConnectionError, asyncio.CancelledError):
            pass
        except Exception as e:
            logger.debug("Webchat: SSE subscriber disconnected: %s", e)
        finally:
            state.unsubscribe(queue)

        return resp

    async def _finalize_stream_state(
        self,
        session_id: str,
        state: _WebchatStreamState,
        handler_task: Optional[asyncio.Task],
    ) -> None:
        """Persist the final assistant response after the gateway task exits."""
        try:
            if handler_task is not None:
                try:
                    await handler_task
                except asyncio.CancelledError:
                    state.error = "Response was cancelled"
                except Exception as e:
                    state.error = str(e)
                    logger.error("Webchat: Gateway handler failed: %s", e)

            if state.error:
                await state.publish({
                    "type": "error",
                    "content": state.error,
                    "session_id": session_id,
                })

            final_content = state.final_content
            if state.pending_media:
                media_suffix = "\n".join(
                    f"MEDIA:{m['path']}" for m in state.pending_media if m.get("path")
                )
                if media_suffix:
                    final_content = (final_content + "\n" + media_suffix).strip()

            final_content = re.sub(
                r'(?m)^MEDIA:\s*(?:<[^>\n]*>|\$\{[^}]*)[^\n]*\n?',
                '',
                final_content,
            ).strip()

            final_message_id = state.final_message_id
            if final_content.strip() or state.final_interactions or state.final_thinking.strip():
                final_message_id = _db_add_message(
                    session_id,
                    "assistant",
                    final_content,
                    interactions=state.final_interactions,
                    thinking=state.final_thinking,
                    blocks=state.final_blocks,
                    sources=state.final_sources,
                    message_id=final_message_id or None,
                )
                state.final_message_id = final_message_id

            state.persisted = True
            if not state.done.is_set():
                await state.publish({
                    "type": "done",
                    "session_id": session_id,
                    "message_id": final_message_id,
                })
        finally:
            # Keep completed state briefly so an already-connected subscriber
            # can receive "done"; durable history is available from SQLite.
            await asyncio.sleep(30)
            if self._active_streams.get(session_id) is state:
                self._active_streams.pop(session_id, None)

    @staticmethod
    def _strip_media_tags(text: str) -> str:
        """Remove MEDIA:<path> tags from display text.
        
        The agent and gateway may embed MEDIA: markers for file delivery,
        but the webchat frontend would render them as broken attachment
        pills since the files aren't served by the webchat file server.
        """
        import re
        text = re.sub(r'''[`\\\"']?MEDIA:\\s*\\S+[`\\\"']?''', "", text).strip()
        # Also strip emoji-prefixed media markers emulated by the agent
        text = re.sub(r'''(?:🖼️\\s*Image:\\s*|🎬\\s*Video:\\s*|🎵\\s*Audio:\\s*|📄\\s*File:\\s*|📎\\s*File:\\s*)\\s*\\S+''', "", text).strip()
        return text

    def _register_interaction(
        self,
        *,
        chat_id: str,
        kind: str,
        title: str,
        content: str,
        controls: list[Dict[str, Any]],
        session_key: str = "",
        state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        interaction_id = str(uuid.uuid4())
        interaction = {
            "id": interaction_id,
            "kind": kind,
            "title": title,
            "content": content,
            "controls": controls,
            "disabled": False,
            "created_at": time.time(),
        }
        self._pending_interactions[interaction_id] = {
            "chat_id": chat_id,
            "session_key": session_key,
            "interaction": interaction,
            "state": state or {},
        }
        return interaction

    async def _send_interaction(
        self,
        *,
        chat_id: str,
        kind: str,
        title: str,
        content: str,
        controls: list[Dict[str, Any]],
        session_key: str = "",
        state: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        interaction = self._register_interaction(
            chat_id=chat_id,
            kind=kind,
            title=title,
            content=content,
            controls=controls,
            session_key=session_key,
            state=state,
        )
        queued = await self._queue_event(chat_id, {
            "type": "interaction",
            "session_id": chat_id,
            "content": content,
            "interaction": interaction,
        })
        if not queued:
            # If there is no active stream, persist the prompt so the user can
            # still see it when the conversation is reopened.
            _db_add_message(chat_id, "assistant", content, interactions=[interaction])
        return SendResult(success=True, message_id=interaction["id"])

    async def send_exec_approval(
        self,
        chat_id: str,
        command: str,
        session_key: str,
        description: str = "dangerous command",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Render a dangerous-command approval prompt with web buttons."""
        cmd_preview = command[:4000] + "..." if len(command) > 4000 else command
        content = (
            "⚠️ **Command Approval Required**\n\n"
            f"```bash\n{cmd_preview}\n```\n"
            f"Reason: {description}"
        )
        controls = [
            {"label": "Allow once", "value": "once", "variant": "default"},
            {"label": "Approve session", "value": "session", "variant": "secondary"},
            {"label": "Always approve", "value": "always", "variant": "secondary"},
            {"label": "Deny", "value": "deny", "variant": "destructive"},
        ]
        return await self._send_interaction(
            chat_id=chat_id,
            kind="exec_approval",
            title="Command approval",
            content=content,
            controls=controls,
            session_key=session_key,
            state={"command": command, "description": description},
        )

    async def send_update_prompt(
        self,
        chat_id: str,
        prompt: str,
        default: str = "",
        session_key: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Render an update-process yes/no prompt."""
        default_hint = f" (default: {default})" if default else ""
        return await self._send_interaction(
            chat_id=chat_id,
            kind="update_prompt",
            title="Update needs your input",
            content=f"⚕ **Update needs your input**\n\n{prompt}{default_hint}",
            controls=[
                {"label": "Yes", "value": "y", "variant": "default"},
                {"label": "No", "value": "n", "variant": "secondary"},
            ],
            session_key=session_key,
            state={"default": default},
        )

    async def send_slash_confirm(
        self,
        chat_id: str,
        title: str,
        message: str,
        session_key: str,
        confirm_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Render slash-command confirmation buttons for webchat."""
        return await self._send_interaction(
            chat_id=chat_id,
            kind="slash_confirm",
            title=title or "Confirm",
            content=message,
            controls=[
                {"label": "Approve once", "value": "once", "variant": "default"},
                {"label": "Always approve", "value": "always", "variant": "secondary"},
                {"label": "Cancel", "value": "cancel", "variant": "destructive"},
            ],
            session_key=session_key,
            state={"confirm_id": confirm_id},
        )

    async def send_model_picker(
        self,
        chat_id: str,
        providers: list,
        current_model: str,
        current_provider: str,
        session_key: str,
        on_model_selected,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Render a provider/model picker backed by web action callbacks."""
        try:
            from hermes_cli.providers import get_label
            provider_label = get_label(current_provider)
        except Exception:
            provider_label = current_provider

        controls = []
        for provider in providers:
            slug = provider.get("slug", "")
            name = provider.get("name", slug)
            count = provider.get("total_models", len(provider.get("models", [])))
            prefix = "✓ " if provider.get("is_current") else ""
            controls.append({
                "label": f"{prefix}{name} ({count})",
                "value": slug,
                "variant": "secondary",
            })
        controls.append({"label": "Cancel", "value": "cancel", "variant": "ghost"})

        content = (
            "⚙ **Model Configuration**\n\n"
            f"Current model: `{current_model or 'unknown'}`\n"
            f"Provider: {provider_label}\n\n"
            "Select a provider:"
        )
        return await self._send_interaction(
            chat_id=chat_id,
            kind="model_provider",
            title="Model configuration",
            content=content,
            controls=controls,
            session_key=session_key,
            state={
                "providers": providers,
                "current_model": current_model,
                "current_provider": current_provider,
                "on_model_selected": on_model_selected,
            },
        )

    async def on_processing_start(self, event: MessageEvent) -> None:
        """Emit a web status event when a turn starts processing."""
        try:
            await self._queue_event(event.source.chat_id, {
                "type": "status",
                "status": "processing",
                "content": "Hermes is processing...",
                "session_id": event.source.chat_id,
            })
        except Exception:
            pass

    async def on_processing_complete(
        self,
        event: MessageEvent,
        outcome: ProcessingOutcome,
    ) -> None:
        """Emit a web status event when a turn completes."""
        try:
            status = "success" if outcome == ProcessingOutcome.SUCCESS else "failure"
            await self._queue_event(event.source.chat_id, {
                "type": "status",
                "status": status,
                "session_id": event.source.chat_id,
            })
        except Exception:
            pass
    
    # ── Route setup ──────────────────────────────────────────────────────
    
    def _setup_routes(self) -> None:
        """Register all HTTP routes on the aiohttp application.
        
        Routes are organised by function:
        - /api/*  — REST API endpoints (authenticated)
        - /*      — Static file serving (SPA)
        """
        if self._app is None:
            return
        
        # Health check (no auth required)
        self._app.router.add_get("/health", self._handle_health)
        
        # Auth endpoint (no auth required)
        self._app.router.add_post("/api/login", self._handle_login)
        
        # Authenticated API endpoints
        self._app.router.add_post("/api/send", self._handle_send)
        self._app.router.add_post("/api/command", self._handle_command)
        self._app.router.add_get("/api/capabilities", self._handle_capabilities)
        self._app.router.add_get("/api/model", self._handle_get_model)
        self._app.router.add_post("/api/model", self._handle_set_model)
        self._app.router.add_post("/api/actions/{action_id}", self._handle_action)

        # User settings
        self._app.router.add_get("/api/settings", self._handle_get_settings)
        self._app.router.add_post("/api/settings", self._handle_post_settings)

        # Session management
        self._app.router.add_get("/api/sessions", self._handle_list_sessions)
        self._app.router.add_get("/api/sessions/slug/{slug}", self._handle_session_by_slug)
        self._app.router.add_get(
            "/api/sessions/{session_id}/messages", self._handle_get_messages
        )
        self._app.router.add_get(
            "/api/sessions/{session_id}/stream", self._handle_session_stream
        )
        self._app.router.add_delete(
            "/api/sessions/{session_id}", self._handle_delete_session
        )
        self._app.router.add_post(
            "/api/sessions/{session_id}/rename", self._handle_rename_session
        )
        self._app.router.add_post(
            "/api/sessions/{session_id}/pin", self._handle_pin_session
        )
        self._app.router.add_post(
            "/api/sessions/{session_id}/archive", self._handle_archive_session
        )

        # File/media serving
        self._app.router.add_get("/api/files/{token}", self._handle_file)
        self._app.router.add_get(
            "/api/files/{session_id}/{filename:.*}", self._handle_files_session
        )
        self._app.router.add_get("/api/media", self._handle_media)
        
        # Static file serving — must be last since it catches all paths
        # We use a custom handler that falls back to index.html for SPA routing
        self._app.router.add_get("/", self._handle_static)
        self._app.router.add_get("/{path:.*}", self._handle_static)
    
    # ── Auth helpers ─────────────────────────────────────────────────────
    
    def _extract_token(self, request: web.Request) -> Optional[str]:
        """Extract a JWT token from the request.
        
        Checks the Authorization header first (Bearer token), then falls
        back to query parameters (for SSE/EventSource compatibility, though
        we use fetch-based streaming so this is a fallback).
        """
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            return auth_header[7:]
        return request.query.get("token", None)
    
    def _require_auth(self, request: web.Request) -> Optional[Dict[str, Any]]:
        """Verify the JWT token on an authenticated request.
        
        Returns the decoded payload if valid, or None if unauthorised.
        The caller should return 401 in that case.
        """
        token = self._extract_token(request)
        if not token:
            return None
        return _verify_jwt(token, self._jwt_secret)
    
    def _json_response(
        self, data: Any, status: int = 200
    ) -> web.Response:
        """Return a JSON response with CORS headers."""
        return web.json_response(data, status=status, headers={
            "Access-Control-Allow-Origin": "*",
        })
    
    # ── HTTP handlers ────────────────────────────────────────────────────
    
    async def _handle_health(self, request: web.Request) -> web.Response:
        """GET /health — simple health check endpoint.
        
        Returns the adapter status so frontend can verify connectivity.
        """
        return self._json_response({
            "status": "ok",
            "platform": "webchat",
            "connected": self.is_connected,
        })
    
    async def _handle_login(self, request: web.Request) -> web.Response:
        """POST /api/login — authenticate a user and return a JWT.
        
        Request body (JSON):
            { "username": "...", "password": "..." }
        
        Response (200):
            { "token": "jwt...", "user": { "username": "...", "user_id": "..." } }
        
        Response (401):
            { "error": "Invalid credentials" }
        """
        try:
            data = await request.json()
        except json.JSONDecodeError:
            return self._json_response({"error": "Invalid JSON"}, status=400)
        
        username = data.get("username", "")
        password = data.get("password", "")
        
        # Verify credentials against the configured values
        if username != self._username or not _verify_password(password, self._password_hash):
            return self._json_response({"error": "Invalid credentials"}, status=401)
        
        # Generate a JWT token for this session
        token = _create_jwt(username, self._jwt_secret)
        
        return self._json_response({
            "token": token,
            "user": {
                "username": username,
                "user_id": f"user-{username}",
            },
        })
    
    async def _handle_send(self, request: web.Request) -> web.Response:
        """POST /api/send — send a message to the agent.
        
        This is the primary interaction endpoint. It:
        1. Verifies the JWT token
        2. Creates a response queue for this exchange
        3. Builds a MessageEvent and dispatches it to the gateway
        4. Returns an SSE stream of response events
        
        Request headers:
            Authorization: Bearer <jwt_token>
        
        Request body (JSON):
            {
                "message": "Hello!",
                "session_id": "optional-uuid"  // omit for new conversation
            }
        
        Response: SSE stream with events:
            data: {"type": "typing"}
            data: {"type": "response", "content": "...", "session_id": "..."}
            data: {"type": "replace", "content": "...", "session_id": "..."}
            data: {"type": "done", "session_id": "..."}
        """
        # Authenticate
        payload = self._require_auth(request)
        if not payload:
            return self._json_response({"error": "Unauthorised"}, status=401)
        
        # Resolve the session id before reading multipart uploads so files can
        # be placed under a stable per-session directory.
        session_id = request.query.get("session_id") or str(uuid.uuid4())
        if request.content_type.startswith("application/json"):
            try:
                data = await request.json()
            except json.JSONDecodeError:
                return self._json_response({"error": "Invalid JSON"}, status=400)
            session_id = data.get("session_id") or session_id
            message_text = data.get("message", "").strip()
            attachments: list[Dict[str, Any]] = []
            media_paths: list[str] = []
            media_types: list[str] = []
        else:
            try:
                message_text, attachments, media_paths, media_types = await _read_uploads(
                    request, session_id
                )
            except Exception as e:
                logger.error("Webchat: Upload parsing failed: %s", e)
                return self._json_response({"error": f"Upload failed: {e}"}, status=400)
        
        if not message_text and not media_paths:
            return self._json_response({"error": "Message or file is required"}, status=400)
        
        active_state = self._active_streams.get(session_id)
        if active_state is not None and active_state.done.is_set() and active_state.persisted:
            self._active_streams.pop(session_id, None)
            active_state = None
        if active_state is not None:
            return self._json_response(
                {"error": "A response is already running for this conversation"},
                status=409,
            )

        # Register the session in the database if new
        _db_add_session(session_id)
        _db_add_message(session_id, "user", message_text, attachments)

        state = _WebchatStreamState(session_id)
        self._active_streams[session_id] = state
        # Reset per-turn state
        self._sources_state.pop(session_id, None)
        
        try:
            # Build a MessageEvent as required by the Hermes gateway
            source = self.build_source(
                chat_id=session_id,
                chat_name="Web Chat",
                chat_type="dm",
                user_id=payload.get("user_id", "unknown"),
                user_name=payload.get("username", "unknown"),
            )
            
            if media_paths:
                has_document = any(not (m or "").startswith("image/") for m in media_types)
                message_type = MessageType.DOCUMENT if has_document else MessageType.PHOTO
                event_text = message_text or _build_upload_placeholder(media_paths, media_types)
            else:
                message_type = MessageType.TEXT
                event_text = message_text
            
            event = MessageEvent(
                text=event_text,
                message_type=message_type,
                source=source,
                message_id=str(uuid.uuid4()),
                media_urls=media_paths,
                media_types=media_types,
            )
            
            # Dispatch the message to the gateway runner. BasePlatformAdapter
            # spawns the real background task and returns quickly, so we track
            # that owner task via the same session key the base adapter uses.
            session_key = build_session_key(
                source,
                group_sessions_per_user=self.config.extra.get("group_sessions_per_user", True),
                thread_sessions_per_user=self.config.extra.get("thread_sessions_per_user", False),
            )
            await self.handle_message(event)
            handler_task = self._session_tasks.get(session_key)
            
            asyncio.create_task(
                self._finalize_stream_state(session_id, state, handler_task)
            )
            return await self._stream_state_to_response(request, state)
        
        except Exception as e:
            logger.error("Webchat: Error handling send request: %s", e)
            if self._active_streams.get(session_id) is state:
                self._active_streams.pop(session_id, None)
            return self._json_response({"error": str(e)}, status=500)

    async def _handle_command(self, request: web.Request) -> web.Response:
        """POST /api/command — send a command into an active webchat run."""
        payload = self._require_auth(request)
        if not payload:
            return self._json_response({"error": "Unauthorised"}, status=401)

        try:
            data = await request.json()
        except json.JSONDecodeError:
            return self._json_response({"error": "Invalid JSON"}, status=400)

        session_id = str(data.get("session_id") or "").strip()
        message_text = str(data.get("message") or "").strip()
        if not session_id or not message_text:
            return self._json_response(
                {"error": "session_id and message are required"},
                status=400,
            )

        state = self._active_streams.get(session_id)
        if state is None or (state.done.is_set() and state.persisted):
            return self._json_response(
                {"error": "No active response for this conversation"},
                status=409,
            )

        if message_text.startswith("/steer "):
            steer_text = message_text[len("/steer "):].strip()
            if steer_text:
                _db_add_message(session_id, "user", steer_text)

        source = self.build_source(
            chat_id=session_id,
            chat_name="Web Chat",
            chat_type="dm",
            user_id=payload.get("user_id", "unknown"),
            user_name=payload.get("username", "unknown"),
        )
        event = MessageEvent(
            text=message_text,
            message_type=MessageType.TEXT,
            source=source,
            message_id=str(uuid.uuid4()),
        )

        await self.handle_message(event)
        return self._json_response({"status": "ok"})
    
    async def _handle_capabilities(self, request: web.Request) -> web.Response:
        """GET /api/capabilities — describe webchat feature support."""
        payload = self._require_auth(request)
        if not payload:
            return self._json_response({"error": "Unauthorised"}, status=401)
        return self._json_response({
            "platform": "webchat",
            "features": {
                "streaming": True,
                "message_editing": True,
                "thinking": True,
                "files": True,
                "images": True,
                "inline_react_artifacts": True,
                "interactions": True,
                "model_picker": True,
                "slash_confirm": True,
                "exec_approval": True,
                "update_prompt": True,
                "tool_call_streaming": True,
            },
        })

    async def _handle_get_model(self, request: web.Request) -> web.Response:
        """GET /api/model — return current model info and available providers.

        Query params:
            session_id (optional) — if provided, checks for session-level model overrides.
        """
        payload = self._require_auth(request)
        if not payload:
            return self._json_response({"error": "Unauthorised"}, status=401)

        try:
            from pathlib import Path
            import yaml
            config_path = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))) / "config.yaml"
            cfg = {}
            if config_path.exists():
                with open(config_path) as f:
                    cfg = yaml.safe_load(f) or {}

            model_cfg = cfg.get("model", {})
            current_model = model_cfg.get("default", "") if isinstance(model_cfg, dict) else ""
            current_provider = model_cfg.get("provider", "openrouter") if isinstance(model_cfg, dict) else "openrouter"
            current_base_url = model_cfg.get("base_url", "") if isinstance(model_cfg, dict) else ""
            current_api_key = model_cfg.get("api_key", "") if isinstance(model_cfg, dict) else ""

            # Check for session-level model override
            session_id = request.query.get("session_id", "").strip()
            if session_id and self._message_handler is not None:
                try:
                    source = self.build_source(
                        chat_id=session_id,
                        chat_name="Web Chat",
                        chat_type="dm",
                        user_id=payload.get("user_id", "unknown"),
                        user_name=payload.get("username", "unknown"),
                    )
                    session_key = build_session_key(
                        source,
                        group_sessions_per_user=self.config.extra.get("group_sessions_per_user", True),
                        thread_sessions_per_user=self.config.extra.get("thread_sessions_per_user", False),
                    )
                    runner = self._message_handler.__self__
                    if hasattr(runner, "_session_model_overrides"):
                        override = runner._session_model_overrides.get(session_key, {})
                        if override:
                            logger.info(
                                "Webchat: session model override found for %s: model=%s provider=%s",
                                session_key[:20], override.get("model"), override.get("provider"),
                            )
                            if override.get("model"):
                                current_model = override["model"]
                            if override.get("provider"):
                                current_provider = override["provider"]
                            if override.get("base_url"):
                                current_base_url = override["base_url"]
                            if override.get("api_key"):
                                current_api_key = override["api_key"]
                except Exception as exc:
                    logger.warning("Webchat: failed to check session overrides: %s", exc)

            user_provs = cfg.get("providers", None)
            custom_provs = cfg.get("custom_providers", None)

            from hermes_cli.model_switch import list_authenticated_providers
            from hermes_cli.providers import get_label

            provider_label = get_label(current_provider)

            providers = list_authenticated_providers(
                current_provider=current_provider,
                current_base_url=current_base_url,
                current_model=current_model,
                user_providers=user_provs,
                custom_providers=custom_provs,
                max_models=50,
            )

            return self._json_response({
                "current_model": current_model,
                "current_provider": current_provider,
                "current_provider_label": provider_label,
                "providers": providers,
            })
        except Exception as e:
            logger.error("Webchat: _handle_get_model failed: %s", e, exc_info=True)
            return self._json_response({"error": str(e)}, status=500)

    async def _handle_set_model(self, request: web.Request) -> web.Response:
        """POST /api/model — switch the model for a webchat session.

        Request body:
            { "provider": "...", "model": "...", "session_id": "..." }
        """
        payload = self._require_auth(request)
        if not payload:
            return self._json_response({"error": "Unauthorised"}, status=401)

        try:
            data = await request.json()
        except json.JSONDecodeError:
            return self._json_response({"error": "Invalid JSON"}, status=400)

        new_model = str(data.get("model", "")).strip()
        new_provider = str(data.get("provider", "")).strip()
        session_id = str(data.get("session_id", "")).strip()

        if not new_model or not new_provider:
            return self._json_response({"error": "Both 'model' and 'provider' are required"}, status=400)

        # We need a session_id to build the session_key for the override.
        # If none provided, generate one so the override can still be stored.
        if not session_id:
            import uuid
            session_id = str(uuid.uuid4())

        try:
            from pathlib import Path
            import yaml
            config_path = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))) / "config.yaml"
            cfg = {}
            if config_path.exists():
                with open(config_path) as f:
                    cfg = yaml.safe_load(f) or {}

            model_cfg = cfg.get("model", {})
            current_model = model_cfg.get("default", "") if isinstance(model_cfg, dict) else ""
            current_provider = model_cfg.get("provider", "openrouter") if isinstance(model_cfg, dict) else "openrouter"
            current_base_url = model_cfg.get("base_url", "") if isinstance(model_cfg, dict) else ""
            current_api_key = model_cfg.get("api_key", "") if isinstance(model_cfg, dict) else ""
            user_provs = cfg.get("providers", None)
            custom_provs = cfg.get("custom_providers", None)

            from hermes_cli.model_switch import switch_model, list_authenticated_providers

            result = switch_model(
                raw_input=new_model,
                current_provider=current_provider,
                current_model=current_model,
                current_base_url=current_base_url,
                current_api_key=current_api_key,
                is_global=False,
                explicit_provider=new_provider,
                user_providers=user_provs,
                custom_providers=custom_provs,
            )

            if not result.success:
                logger.warning("Webchat: model switch failed: %s", result.error_message)
                return self._json_response({"error": result.error_message}, status=400)

            logger.info(
                "Webchat: switching model to %s on provider %s (session %s)",
                result.new_model, result.target_provider, session_id[:20],
            )

            # Build the session key matching what _handle_send uses
            source = self.build_source(
                chat_id=session_id,
                chat_name="Web Chat",
                chat_type="dm",
                user_id=payload.get("user_id", "unknown"),
                user_name=payload.get("username", "unknown"),
            )
            session_key = build_session_key(
                source,
                group_sessions_per_user=self.config.extra.get("group_sessions_per_user", True),
                thread_sessions_per_user=self.config.extra.get("thread_sessions_per_user", False),
            )

            # Access the gateway runner through the bound _message_handler
            runner_updated = False
            if self._message_handler is not None:
                try:
                    runner = self._message_handler.__self__
                    # Update the session model override
                    if hasattr(runner, "_session_model_overrides"):
                        runner._session_model_overrides[session_key] = {
                            "model": result.new_model,
                            "provider": result.target_provider,
                            "api_key": result.api_key,
                            "base_url": result.base_url,
                            "api_mode": result.api_mode,
                        }
                        runner_updated = True
                        logger.info("Webchat: stored session override for %s", session_key[:24])

                    # Update cached agent in-place
                    if hasattr(runner, "_agent_cache_lock") and hasattr(runner, "_agent_cache"):
                        with runner._agent_cache_lock:
                            cached_entry = runner._agent_cache.get(session_key)
                        if cached_entry and cached_entry[0] is not None:
                            try:
                                cached_entry[0].switch_model(
                                    new_model=result.new_model,
                                    new_provider=result.target_provider,
                                    api_key=result.api_key,
                                    base_url=result.base_url,
                                    api_mode=result.api_mode,
                                )
                            except Exception as exc:
                                logger.warning("Webchat: model switch failed for cached agent: %s", exc)

                        # Evict cached agent so next turn creates a fresh one from override
                        if hasattr(runner, "_evict_cached_agent"):
                            runner._evict_cached_agent(session_key)

                except Exception as exc:
                    logger.warning("Webchat: failed to update runner state: %s", exc)

            # Build confirmation text
            from hermes_cli.model_switch import resolve_display_context_length
            plabel = result.provider_label or result.target_provider
            lines = [f"Model switched to `{result.new_model}`"]
            lines.append(f"Provider: {plabel}")
            mi = result.model_info
            sw_config_ctx = None
            try:
                sw_cfg_path = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))) / "config.yaml"
                if sw_cfg_path.exists():
                    with open(sw_cfg_path) as f:
                        sw_cfg = yaml.safe_load(f) or {}
                    sw_model_cfg = sw_cfg.get("model", {})
                    if isinstance(sw_model_cfg, dict):
                        sw_raw = sw_model_cfg.get("context_length")
                        if sw_raw is not None:
                            sw_config_ctx = int(sw_raw)
            except Exception:
                pass
            ctx = resolve_display_context_length(
                result.new_model,
                result.target_provider,
                base_url=result.base_url or current_base_url or "",
                api_key=result.api_key or current_api_key or "",
                model_info=mi,
                custom_providers=custom_provs,
                config_context_length=sw_config_ctx,
            )
            if ctx:
                lines.append(f"Context: {ctx:,} tokens")
            if mi:
                if mi.max_output:
                    lines.append(f"Max output: {mi.max_output:,} tokens")
                if mi.has_cost_data():
                    lines.append(f"Cost: {mi.format_cost()}")
                lines.append(f"Capabilities: {mi.format_capabilities()}")
            lines.append("_(session only)_")

            return self._json_response({
                "status": "ok",
                "message": "\n".join(lines),
                "model": result.new_model,
                "provider": result.target_provider,
                "provider_label": plabel,
            })
        except Exception as e:
            logger.error("Webchat: _handle_set_model failed: %s", e, exc_info=True)
            return self._json_response({"error": str(e)}, status=500)

    async def _handle_action(self, request: web.Request) -> web.Response:
        """POST /api/actions/{action_id} — resolve a web interaction button."""
        payload = self._require_auth(request)
        if not payload:
            return self._json_response({"error": "Unauthorised"}, status=401)

        action_id = request.match_info.get("action_id", "")
        entry = self._pending_interactions.get(action_id)
        if not entry:
            return self._json_response({"error": "Action expired or already handled"}, status=404)

        try:
            data = await request.json()
        except json.JSONDecodeError:
            data = {}
        value = str(data.get("value", "")).strip()
        interaction = entry["interaction"]
        kind = interaction.get("kind")
        chat_id = entry.get("chat_id", "")
        state = entry.get("state") or {}
        session_key = entry.get("session_key", "")

        try:
            if kind == "exec_approval":
                if value not in {"once", "session", "always", "deny"}:
                    return self._json_response({"error": "Invalid approval choice"}, status=400)
                from tools.approval import resolve_gateway_approval

                resolved = resolve_gateway_approval(session_key, value)
                interaction["disabled"] = True
                interaction["selected"] = value
                _db_update_interaction(action_id, interaction)
                self._pending_interactions.pop(action_id, None)
                return self._json_response({
                    "status": "ok",
                    "resolved": resolved,
                    "interaction": interaction,
                    "message": None,
                })

            if kind == "slash_confirm":
                if value not in {"once", "always", "cancel"}:
                    return self._json_response({"error": "Invalid confirmation choice"}, status=400)
                from tools import slash_confirm as _slash_confirm_mod

                result_text = await _slash_confirm_mod.resolve(
                    session_key,
                    str(state.get("confirm_id", "")),
                    value,
                )
                interaction["disabled"] = True
                interaction["selected"] = value
                _db_update_interaction(action_id, interaction)
                self._pending_interactions.pop(action_id, None)
                message = None
                if result_text:
                    message_id = _db_add_message(chat_id, "assistant", result_text)
                    message = {
                        "id": message_id,
                        "role": "assistant",
                        "content": result_text,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                return self._json_response({
                    "status": "ok",
                    "interaction": interaction,
                    "message": message,
                })

            if kind == "update_prompt":
                if value not in {"y", "n"}:
                    return self._json_response({"error": "Invalid update response"}, status=400)
                from hermes_constants import get_hermes_home

                response_path = get_hermes_home() / ".update_response"
                tmp_path = response_path.with_suffix(".tmp")
                tmp_path.write_text(value)
                tmp_path.replace(response_path)
                interaction["disabled"] = True
                interaction["selected"] = value
                _db_update_interaction(action_id, interaction)
                self._pending_interactions.pop(action_id, None)
                return self._json_response({
                    "status": "ok",
                    "interaction": interaction,
                    "message": f"Update prompt answered: {'Yes' if value == 'y' else 'No'}.",
                })

            if kind == "model_provider":
                if value == "cancel":
                    interaction["disabled"] = True
                    interaction["selected"] = "cancel"
                    _db_update_interaction(action_id, interaction)
                    self._pending_interactions.pop(action_id, None)
                    return self._json_response({
                        "status": "ok",
                        "interaction": interaction,
                        "message": "Model selection cancelled.",
                    })
                return self._model_picker_provider_response(action_id, entry, value, 0)

            if kind == "model_select":
                return await self._model_picker_model_response(action_id, entry, value)

            # ── Clarify response ──────────────────────────────────────────
            if kind == "clarify_response":
                self._pending_clarify_values[action_id] = value
                ev = self._pending_clarify.pop(action_id, None)
                if ev and not ev.is_set():
                    ev.set()
                interaction["disabled"] = True
                interaction["selected"] = value
                _db_update_interaction(action_id, interaction)
                self._pending_interactions.pop(action_id, None)
                return self._json_response({
                    "status": "ok",
                    "interaction": interaction,
                    "message": None,
                })

            return self._json_response({"error": "Unsupported action type"}, status=400)
        except Exception as e:
            logger.error("Webchat: action %s failed: %s", action_id, e, exc_info=True)
            return self._json_response({"error": str(e)}, status=500)

    def _model_picker_provider_response(
        self,
        action_id: str,
        entry: Dict[str, Any],
        provider_slug: str,
        page: int,
    ) -> web.Response:
        """Update a model picker interaction to show models for one provider."""
        state = entry.get("state") or {}
        provider = next(
            (p for p in state.get("providers", []) if p.get("slug") == provider_slug),
            None,
        )
        if not provider:
            return self._json_response({"error": "Provider not found"}, status=404)

        models = provider.get("models", [])
        page_size = 10
        total_pages = max(1, (len(models) + page_size - 1) // page_size)
        page = max(0, min(page, total_pages - 1))
        start = page * page_size
        end = min(start + page_size, len(models))

        controls: list[Dict[str, Any]] = []
        for index, model_id in enumerate(models[start:end], start=start):
            label = model_id.split("/")[-1] if "/" in model_id else model_id
            if len(label) > 54:
                label = label[:51] + "..."
            controls.append({"label": label, "value": str(index), "variant": "secondary"})
        if total_pages > 1:
            if page > 0:
                controls.append({"label": "Prev", "value": f"page:{page - 1}", "variant": "ghost"})
            controls.append({"label": f"{page + 1}/{total_pages}", "value": "noop", "variant": "ghost", "disabled": True})
            if page < total_pages - 1:
                controls.append({"label": "Next", "value": f"page:{page + 1}", "variant": "ghost"})
        controls.append({"label": "Back", "value": "back", "variant": "ghost"})
        controls.append({"label": "Cancel", "value": "cancel", "variant": "ghost"})

        total = provider.get("total_models", len(models))
        shown = len(models)
        extra = f"\n\n_{total - shown} more available - type `/model <name>` directly._" if total > shown else ""
        interaction = entry["interaction"]
        interaction.update({
            "kind": "model_select",
            "title": f"Models for {provider.get('name', provider_slug)}",
            "content": (
                "⚙ **Model Configuration**\n\n"
                f"Provider: **{provider.get('name', provider_slug)}**\n"
                f"Select a model:{extra}"
            ),
            "controls": controls,
        })
        state["selected_provider"] = provider_slug
        state["model_page"] = page
        entry["state"] = state
        self._pending_interactions[action_id] = entry
        _db_update_interaction(action_id, interaction)
        return self._json_response({"status": "ok", "interaction": interaction})

    async def _model_picker_model_response(
        self,
        action_id: str,
        entry: Dict[str, Any],
        value: str,
    ) -> web.Response:
        """Resolve a model picker model action."""
        state = entry.get("state") or {}
        interaction = entry["interaction"]
        provider_slug = state.get("selected_provider", "")
        if value.startswith("page:"):
            try:
                page = int(value.split(":", 1)[1])
            except ValueError:
                page = 0
            return self._model_picker_provider_response(action_id, entry, provider_slug, page)
        if value == "back":
            interaction.update({
                "kind": "model_provider",
                "title": "Model configuration",
                "content": (
                    "⚙ **Model Configuration**\n\n"
                    f"Current model: `{state.get('current_model') or 'unknown'}`\n\n"
                    "Select a provider:"
                ),
                "controls": [
                    {
                        "label": f"{'✓ ' if p.get('is_current') else ''}{p.get('name', p.get('slug'))} ({p.get('total_models', len(p.get('models', [])))})",
                        "value": p.get("slug", ""),
                        "variant": "secondary",
                    }
                    for p in state.get("providers", [])
                ] + [{"label": "Cancel", "value": "cancel", "variant": "ghost"}],
            })
            _db_update_interaction(action_id, interaction)
            return self._json_response({"status": "ok", "interaction": interaction})
        if value == "cancel":
            interaction["disabled"] = True
            interaction["selected"] = "cancel"
            _db_update_interaction(action_id, interaction)
            self._pending_interactions.pop(action_id, None)
            return self._json_response({
                "status": "ok",
                "interaction": interaction,
                "message": "Model selection cancelled.",
            })
        if value == "noop":
            return self._json_response({"status": "ok", "interaction": interaction})

        provider = next(
            (p for p in state.get("providers", []) if p.get("slug") == provider_slug),
            None,
        )
        models = provider.get("models", []) if provider else []
        try:
            index = int(value)
        except ValueError:
            return self._json_response({"error": "Invalid model selection"}, status=400)
        if index < 0 or index >= len(models):
            return self._json_response({"error": "Model selection out of range"}, status=400)

        callback = state.get("on_model_selected")
        if not callback:
            return self._json_response({"error": "Model picker callback expired"}, status=410)
        model_id = models[index]
        result_text = await callback(entry.get("chat_id", ""), model_id, provider_slug)
        interaction["disabled"] = True
        interaction["selected"] = model_id
        _db_update_interaction(action_id, interaction)
        self._pending_interactions.pop(action_id, None)
        message_id = _db_add_message(entry.get("chat_id", ""), "assistant", result_text)
        return self._json_response({
            "status": "ok",
            "interaction": interaction,
            "message": {
                "id": message_id,
                "role": "assistant",
                "content": result_text,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        })

    async def _handle_reaction(self, request: web.Request) -> web.Response:
        """POST /api/messages/{message_id}/reactions — toggle a message reaction."""
        payload = self._require_auth(request)
        if not payload:
            return self._json_response({"error": "Unauthorised"}, status=401)
        try:
            data = await request.json()
        except json.JSONDecodeError:
            data = {}
        reaction = str(data.get("reaction", "")).strip()
        if reaction not in {"thumbs_up", "thumbs_down", "heart", "laugh", "eyes"}:
            return self._json_response({"error": "Unsupported reaction"}, status=400)
        try:
            reactions = _db_set_reaction(
                request.match_info.get("message_id", ""),
                reaction,
                payload.get("user_id") or payload.get("username") or "webchat",
            )
            return self._json_response({"reactions": reactions})
        except KeyError:
            return self._json_response({"error": "Message not found"}, status=404)
        except Exception as e:
            return self._json_response({"error": str(e)}, status=500)
    
    async def _handle_list_sessions(self, request: web.Request) -> web.Response:
        """GET /api/sessions — list all conversation sessions.
        
        Query params:
            filter (optional) — "archived" to show archived only.
        
        Returns sessions ordered by most recently active first,
        with pinned sessions at the top.
        """
        payload = self._require_auth(request)
        if not payload:
            return self._json_response({"error": "Unauthorised"}, status=401)
        
        try:
            filter_by = request.query.get("filter", "").strip().lower()
            sessions = _db_get_sessions(filter_by=filter_by)
            return self._json_response({"sessions": sessions})
        except Exception as e:
            return self._json_response({"error": str(e)}, status=500)
    
    async def _handle_session_by_slug(self, request: web.Request) -> web.Response:
        """GET /api/sessions/slug/{slug} — look up a session by its slug."""
        payload = self._require_auth(request)
        if not payload:
            return self._json_response({"error": "Unauthorised"}, status=401)
        
        slug = request.match_info.get("slug", "")
        if not slug:
            return self._json_response({"error": "Slug required"}, status=400)
        
        try:
            sessions = _db_get_sessions()
            for s in sessions:
                if s.get("slug") == slug:
                    return self._json_response({"session": s})
            return self._json_response({"error": "Session not found"}, status=404)
        except Exception as e:
            return self._json_response({"error": str(e)}, status=500)
    
    async def _handle_get_messages(self, request: web.Request) -> web.Response:
        """GET /api/sessions/{session_id}/messages — load persisted messages."""
        payload = self._require_auth(request)
        if not payload:
            return self._json_response({"error": "Unauthorised"}, status=401)
        
        session_id = request.match_info.get("session_id", "")
        if not session_id:
            return self._json_response({"error": "Session ID required"}, status=400)
        
        try:
            return self._json_response({"messages": _db_get_messages(session_id)})
        except Exception as e:
            return self._json_response({"error": str(e)}, status=500)

    async def _handle_session_stream(self, request: web.Request) -> web.StreamResponse:
        """GET /api/sessions/{session_id}/stream — attach to a live response."""
        payload = self._require_auth(request)
        if not payload:
            return self._json_response({"error": "Unauthorised"}, status=401)

        session_id = request.match_info.get("session_id", "")
        if not session_id:
            return self._json_response({"error": "Session ID required"}, status=400)

        state = self._active_streams.get(session_id)
        if state is None:
            return web.Response(status=204, headers={"Access-Control-Allow-Origin": "*"})
        if state.done.is_set() and state.persisted:
            return web.Response(status=204, headers={"Access-Control-Allow-Origin": "*"})

        return await self._stream_state_to_response(request, state)
    
    async def _handle_delete_session(
        self, request: web.Request
    ) -> web.Response:
        """DELETE /api/sessions/{session_id} — delete a session."""
        payload = self._require_auth(request)
        if not payload:
            return self._json_response({"error": "Unauthorised"}, status=401)
        
        session_id = request.match_info.get("session_id", "")
        if not session_id:
            return self._json_response({"error": "Session ID required"}, status=400)
        
        try:
            deleted = _db_delete_session(session_id)
            if deleted:
                return self._json_response({"status": "deleted"})
            else:
                return self._json_response({"error": "Session not found"}, status=404)
        except Exception as e:
            return self._json_response({"error": str(e)}, status=500)
    
    async def _handle_file(self, request: web.Request) -> web.StreamResponse:
        """GET /api/files/{token} — serve an uploaded file back to the UI."""
        payload = self._require_auth(request)
        if not payload:
            return self._json_response({"error": "Unauthorised"}, status=401)
        
        decoded = _decode_file_token(request.match_info.get("token", ""))
        if not decoded:
            return web.Response(status=404, text="Not found")
        session_id, filename = decoded
        file_path = os.path.join(_UPLOAD_ROOT, session_id, filename)
        real_path = os.path.realpath(file_path)
        real_root = os.path.realpath(_UPLOAD_ROOT)
        if not real_path.startswith(real_root) or not os.path.isfile(real_path):
            return web.Response(status=404, text="Not found")
        return web.FileResponse(real_path)
    
    async def _handle_files_session(self, request: web.Request) -> web.StreamResponse:
        """GET /api/files/{session_id}/{filename} — serve uploaded files with JWT auth.

        Files are served from ~/.hermes/webchat_uploads/{session_id}/{filename}.
        """
        payload = self._require_auth(request)
        if not payload:
            return self._json_response({"error": "Unauthorised"}, status=401)

        session_id = request.match_info.get("session_id", "")
        filename = request.match_info.get("filename", "")
        if not session_id or not filename:
            return web.Response(status=404, text="Not found")

        # Prevent path traversal
        safe_filename = _safe_filename(filename)
        safe_session = os.path.basename(os.path.normpath(session_id))
        file_path = os.path.join(_UPLOAD_ROOT, safe_session, safe_filename)
        real_path = os.path.realpath(file_path)
        real_root = os.path.realpath(_UPLOAD_ROOT)
        if not real_path.startswith(real_root) or not os.path.isfile(real_path):
            return web.Response(status=404, text="Not found")
        return web.FileResponse(real_path)

    async def _handle_media(self, request: web.Request) -> web.StreamResponse:
        """GET /api/media — proxy a local file to the frontend with JWT auth.

        Query params:
            path (required) — absolute path to a local file (e.g. /tmp/...)
        """
        payload = self._require_auth(request)
        if not payload:
            return self._json_response({"error": "Unauthorised"}, status=401)

        file_path = request.query.get("path", "").strip()
        if not file_path:
            return self._json_response({"error": "Missing 'path' query parameter"}, status=400)

        real_path = os.path.realpath(file_path)
        if not os.path.isfile(real_path):
            return self._json_response({"error": "File not found"}, status=404)

        # Basic security: require an absolute path, reject if it looks like
        # an attempt to read system files outside of /tmp or /home
        if not os.path.isabs(real_path):
            return self._json_response({"error": "Path must be absolute"}, status=400)

        # Serve with Content-Disposition for non-image files so they download
        # rather than rendering inline (especially for code/docs/archive types).
        filename = os.path.basename(real_path)
        ext = os.path.splitext(filename)[1].lower()
        _DISPOSITION_INLINE = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.svg',
                               '.mp4', '.mov', '.webm', '.ogv', '.mp3', '.ogg', '.wav',
                               '.m4a', '.flac', '.opus', '.pdf'}
        if ext in _DISPOSITION_INLINE:
            return web.FileResponse(real_path)
        return web.FileResponse(
            real_path,
            headers={
                "Content-Disposition": f'attachment; filename="{_safe_filename(filename)}"',
            },
        )

    # ── Settings handlers ───────────────────────────────────────────────

    async def _handle_get_settings(self, request: web.Request) -> web.Response:
        """GET /api/settings — return the logged-in user's settings."""
        payload = self._require_auth(request)
        if not payload:
            return self._json_response({"error": "Unauthorised"}, status=401)

        user_id = payload.get("user_id", "")
        if not user_id:
            return self._json_response({"error": "Invalid user"}, status=400)

        try:
            settings = _db_get_user_settings(user_id)
            return self._json_response(settings)
        except Exception as e:
            return self._json_response({"error": str(e)}, status=500)

    async def _handle_post_settings(self, request: web.Request) -> web.Response:
        """POST /api/settings — save user settings.

        Request body (JSON):
            { "accent_color"?: "...", "bg_color"?: "...", "theme"?: "dark"|"light" }
        """
        payload = self._require_auth(request)
        if not payload:
            return self._json_response({"error": "Unauthorised"}, status=401)

        user_id = payload.get("user_id", "")
        if not user_id:
            return self._json_response({"error": "Invalid user"}, status=400)

        try:
            data = await request.json()
        except json.JSONDecodeError:
            return self._json_response({"error": "Invalid JSON"}, status=400)

        # Validate theme if provided
        theme = data.get("theme", "")
        if theme and theme not in ("dark", "light"):
            return self._json_response({"error": "theme must be 'dark' or 'light'"}, status=400)

        try:
            settings = _db_set_user_settings(user_id, data)
            return self._json_response(settings)
        except Exception as e:
            return self._json_response({"error": str(e)}, status=500)

    # ── Session management handlers ─────────────────────────────────────

    async def _handle_rename_session(self, request: web.Request) -> web.Response:
        """POST /api/sessions/{session_id}/rename — rename a session.

        Request body (JSON):
            { "title": "New title" }
        """
        payload = self._require_auth(request)
        if not payload:
            return self._json_response({"error": "Unauthorised"}, status=401)

        session_id = request.match_info.get("session_id", "")
        if not session_id:
            return self._json_response({"error": "Session ID required"}, status=400)

        try:
            data = await request.json()
        except json.JSONDecodeError:
            return self._json_response({"error": "Invalid JSON"}, status=400)

        title = str(data.get("title", "")).strip()
        if not title:
            return self._json_response({"error": "Title is required"}, status=400)

        try:
            updated = _db_rename_session(session_id, title)
            if updated:
                return self._json_response({"status": "ok", "title": title.strip()[:60]})
            else:
                return self._json_response({"error": "Session not found"}, status=404)
        except Exception as e:
            return self._json_response({"error": str(e)}, status=500)

    async def _handle_pin_session(self, request: web.Request) -> web.Response:
        """POST /api/sessions/{session_id}/pin — toggle pinned status."""
        payload = self._require_auth(request)
        if not payload:
            return self._json_response({"error": "Unauthorised"}, status=401)

        session_id = request.match_info.get("session_id", "")
        if not session_id:
            return self._json_response({"error": "Session ID required"}, status=400)

        try:
            updated = _db_toggle_pin(session_id)
            if updated:
                return self._json_response({"status": "ok"})
            else:
                return self._json_response({"error": "Session not found"}, status=404)
        except Exception as e:
            return self._json_response({"error": str(e)}, status=500)

    async def _handle_archive_session(self, request: web.Request) -> web.Response:
        """POST /api/sessions/{session_id}/archive — toggle archived status."""
        payload = self._require_auth(request)
        if not payload:
            return self._json_response({"error": "Unauthorised"}, status=401)

        session_id = request.match_info.get("session_id", "")
        if not session_id:
            return self._json_response({"error": "Session ID required"}, status=400)

        try:
            updated = _db_toggle_archive(session_id)
            if updated:
                return self._json_response({"status": "ok"})
            else:
                return self._json_response({"error": "Session not found"}, status=404)
        except Exception as e:
            return self._json_response({"error": str(e)}, status=500)

    async def _handle_static(self, request: web.Request) -> web.Response:
        """Serve static files for the SPA.
        
        If the requested file doesn't exist, fall back to index.html.
        This enables client-side routing in the React app (paths like
        /chat, /settings, etc. all served by index.html).
        """
        # Determine the file path
        path = request.match_info.get("path", "")
        if not path or path == "/":
            file_path = os.path.join(self._spa_dir, "index.html")
        else:
            file_path = os.path.join(self._spa_dir, path)
        
        # Security: prevent directory traversal
        real_path = os.path.realpath(file_path)
        real_spa_dir = os.path.realpath(self._spa_dir)
        if not real_path.startswith(real_spa_dir):
            return web.Response(status=404, text="Not found")
        
        # Serve the file if it exists, otherwise fall back to index.html
        if os.path.isfile(real_path):
            return web.FileResponse(real_path)
        else:
            # SPA fallback: all unrecognised routes serve index.html
            index_path = os.path.join(self._spa_dir, "index.html")
            if os.path.isfile(index_path):
                return web.FileResponse(index_path)
            else:
                return web.Response(
                    status=503,
                    text="SPA not built. Run 'npm run build' in the chat-hermes directory.",
                )


# ---------------------------------------------------------------------------
# Plugin registration functions
# ---------------------------------------------------------------------------

def check_requirements() -> bool:
    """Check if the webchat platform can start.
    
    All required dependencies (aiohttp, bcrypt, pyjwt) are installed.
    The env vars must be configured for the platform to work.
    """
    username = os.environ.get("WEBCHAT_USERNAME", "")
    password = os.environ.get("WEBCHAT_PASSWORD", "")
    jwt_secret = os.environ.get("WEBCHAT_JWT_SECRET", "")
    return bool(username and password and jwt_secret)


def validate_config(config) -> bool:
    """Validate that the platform configuration is complete.
    
    Checks both env vars and config.yaml extra dict for required settings.
    """
    extra = getattr(config, "extra", {}) or {}
    username = os.environ.get("WEBCHAT_USERNAME") or extra.get("username", "")
    password = os.environ.get("WEBCHAT_PASSWORD") or extra.get("password", "")
    jwt_secret = os.environ.get("WEBCHAT_JWT_SECRET") or extra.get("jwt_secret", "")
    return bool(username and password and jwt_secret)


def register(ctx):
    """Plugin entry point — called by the Hermes plugin system.
    
    This function tells Hermes about the webchat platform, including how
    to create the adapter, what config it needs, and how it integrates
    with the gateway's auth and delivery systems.
    """
    ctx.register_platform(
        # Platform identity
        name="webchat",
        label="Web Chat",
        adapter_factory=lambda cfg: WebchatAdapter(cfg),
        
        # Requirements and validation
        check_fn=check_requirements,
        validate_config=validate_config,
        required_env=["WEBCHAT_USERNAME", "WEBCHAT_PASSWORD", "WEBCHAT_JWT_SECRET"],
        install_hint="No extra packages needed (aiohttp, bcrypt, PyJWT bundled with gateway)",
        
        # Auth env vars for _is_user_authorized() integration
        allowed_users_env="WEBCHAT_ALLOWED_USERS",
        allow_all_env="WEBCHAT_ALLOW_ALL_USERS",
        
        # Message length limit (0 = no limit — web SPA can handle large messages)
        max_message_length=0,
        
        # Display options
        emoji="🌐",
        pii_safe=False,
        allow_update_command=True,
        
        # LLM guidance injected into system prompt
        platform_hint=(
            "You are chatting via a Web Chat interface. "
            "This platform supports GitHub Flavored Markdown including tables, "
            "task lists, strikethrough, links, images, syntax-highlighted code "
            "blocks, inline LaTeX with $...$, and block LaTeX with $$...$$. "
            "For flowcharts and diagrams, use Mermaid fenced code blocks with "
            "```mermaid, ```flowchart, or ```diagram; these render natively in "
            "the web UI. For interactive artifacts, use explicit artifact fences "
            "only: ```artifact-react for React/TSX, ```artifact-html for HTML, "
            "or ```artifact for a React artifact. Do not use plain ```tsx, "
            "```jsx, ```react, or ```html when you intend an artifact; those are "
            "shown as normal code. React artifacts run in a sandbox with preview, "
            "source, fullscreen, copy, and sidebar controls. They include a small "
            "shadcn-style component kit; you may import components from "
            "@/components/ui/button, @/components/ui/card, @/components/ui/badge, "
            "@/components/ui/input, @/components/ui/textarea, or @/components/ui/tabs, "
            "or access them via the Shadcn global. Recharts, d3, React hooks, and "
            "basic lucide-react icon imports are also available. Prefer Mermaid "
            "for static diagrams and artifacts only for interactive UI, charts, "
            "or demos. Raw inline HTML in normal markdown is not rendered; place "
            "HTML in an artifact-html fence when needed."
        ),
    )
