"""case_service.py - Case Management (Phase 11)
MongoDB-backed case CRUD for courtRoom.ai
"""
import os
from datetime import datetime
from typing import Dict, List, Optional

from dotenv import load_dotenv
from pymongo import MongoClient
from bson.objectid import ObjectId

load_dotenv(override=True)

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "courtroom_ai")

VALID_STATUSES = ["draft", "active", "pending", "closed"]


class CaseService:
    def __init__(self):
        self.db = MongoClient(MONGODB_URI)[MONGODB_DB]
        self.cases = self.db.cases
        self.cases.create_index([("user_id", 1), ("created_at", -1)])

    def create_case(self, user_id: str, client_name: str, case_type: str,
                    description: str = "", status: str = "draft",
                    metadata: Optional[Dict] = None) -> Optional[Dict]:
        document = {
            "user_id": user_id,
            "client_name": client_name,
            "case_type": case_type,
            "description": description,
            "status": status if status in VALID_STATUSES else "draft",
            "evidence_count": 0,
            "metadata": metadata or {},
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        result = self.cases.insert_one(document)
        return self._serialize({"_id": result.inserted_id, **document})

    def get_case(self, case_id: str) -> Optional[Dict]:
        try:
            doc = self.cases.find_one({"_id": ObjectId(case_id)})
        except Exception:
            return None
        return self._serialize(doc) if doc else None

    def list_cases(self, user_id: Optional[str] = None,
                   status: Optional[str] = None, limit: int = 50) -> List[Dict]:
        query = {}
        if user_id:
            query["user_id"] = user_id
        if status:
            query["status"] = status
        docs = list(self.cases.find(query).sort("created_at", -1).limit(limit))
        return [self._serialize(d) for d in docs]

    def update_case_status(self, case_id: str, status: str) -> Optional[Dict]:
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status. Valid: {VALID_STATUSES}")
        result = self.cases.update_one(
            {"_id": ObjectId(case_id)},
            {"$set": {"status": status, "updated_at": datetime.now()}}
        )
        if result.matched_count == 0:
            return None
        return self.get_case(case_id)

    def increment_evidence_count(self, case_id: str) -> None:
        self.cases.update_one(
            {"_id": ObjectId(case_id)},
            {"$inc": {"evidence_count": 1}, "$set": {"updated_at": datetime.now()}}
        )

    def delete_case(self, case_id: str) -> bool:
        result = self.cases.delete_one({"_id": ObjectId(case_id)})
        return result.deleted_count > 0

    @staticmethod
    def _serialize(doc: Dict) -> Dict:
        doc["_id"] = str(doc["_id"])
        doc["created_at"] = doc["created_at"].isoformat() if doc.get("created_at") else None
        doc["updated_at"] = doc["updated_at"].isoformat() if doc.get("updated_at") else None
        return doc
