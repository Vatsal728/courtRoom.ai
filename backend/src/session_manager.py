"""
session_manager.py - Persistent Multi-Turn Tool Session Store for courtRoom.ai.
Uses MongoDB (db.tool_sessions) with native TTL indexing so user conversations survive
server restarts, multi-worker uvicorn processes, and auto-expire after 15 minutes.
Gracefully falls back to an in-memory dictionary if MongoDB is temporarily unavailable.
"""
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv(override=True)

_DEFAULT_TTL_SECONDS = int(os.getenv("TOOL_SESSION_TTL_SECONDS", "900"))  # 15 minutes
_IN_MEMORY_FALLBACK: Dict[str, Dict[str, Any]] = {}


class ToolSessionManager:
    """Manages multi-turn tool parameter accumulation in MongoDB with TTL."""

    def __init__(self, mongo_uri: Optional[str] = None, db_name: Optional[str] = None):
        self.uri = mongo_uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        self.db_name = db_name or os.getenv("MONGODB_DB", "courtroom_ai")
        self.client = None
        self.db = None
        self.collection = None
        self._init_mongo()

    def _init_mongo(self):
        try:
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=2000)
            self.db = self.client[self.db_name]
            self.collection = self.db["tool_sessions"]
            # Ensure TTL index on expire_at
            self.collection.create_index("expire_at", expireAfterSeconds=0)
            self.collection.create_index("user_id", unique=True)
        except Exception:
            self.collection = None

    def store_session(
        self,
        user_id: str,
        tool_name: str,
        params: Dict[str, Any],
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
    ) -> None:
        if not user_id:
            return
        now = datetime.now(timezone.utc)
        expire_at = now + timedelta(seconds=ttl_seconds)
        clean_params = dict(params or {})
        doc = {
            "user_id": user_id,
            "tool_name": tool_name,
            "params": clean_params,
            "updated_at": now,
            "expire_at": expire_at,
        }
        # Always keep in-memory fallback synchronized in case MongoDB drops mid-session
        _IN_MEMORY_FALLBACK[user_id] = {**doc, "ts": time.monotonic()}
        if self.collection is not None:
            try:
                self.collection.update_one(
                    {"user_id": user_id},
                    {"$set": doc},
                    upsert=True,
                )
            except Exception:
                pass

    def get_session(self, user_id: str) -> Optional[Dict[str, Any]]:
        if not user_id:
            return None
        if self.collection is not None:
            try:
                res = self.collection.find_one({"user_id": user_id})
                if res:
                    expire_at = res.get("expire_at")
                    if isinstance(expire_at, datetime):
                        exp_utc = expire_at if expire_at.tzinfo else expire_at.replace(tzinfo=timezone.utc)
                        if datetime.now(timezone.utc) > exp_utc:
                            self.clear_session(user_id)
                            return None
                    return {
                        "name": res.get("tool_name"),
                        "params": dict(res.get("params") or {}),
                        "updated_at": res.get("updated_at"),
                    }
            except Exception:
                pass
        # Fallback in-memory check
        item = _IN_MEMORY_FALLBACK.get(user_id)
        if item:
            if time.monotonic() - item.get("ts", 0) > _DEFAULT_TTL_SECONDS:
                _IN_MEMORY_FALLBACK.pop(user_id, None)
                return None
            return {
                "name": item.get("tool_name"),
                "params": dict(item.get("params") or {}),
                "updated_at": item.get("updated_at"),
            }
        return None

    def clear_session(self, user_id: str) -> None:
        if not user_id:
            return
        _IN_MEMORY_FALLBACK.pop(user_id, None)
        if self.collection is not None:
            try:
                self.collection.delete_one({"user_id": user_id})
            except Exception:
                pass


_session_manager = ToolSessionManager()


def get_session_manager() -> ToolSessionManager:
    return _session_manager
