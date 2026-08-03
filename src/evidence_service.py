"""evidence_service.py - Evidence Upload via GridFS (Phase 12)
Stores case evidence files in MongoDB GridFS with metadata
"""
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from gridfs import GridFS
from pymongo import MongoClient
from bson.objectid import ObjectId

load_dotenv(override=True)

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "courtroom_ai")


class EvidenceService:
    def __init__(self, case_service):
        self.db = MongoClient(MONGODB_URI)[MONGODB_DB]
        self.fs = GridFS(self.db, collection="evidence_files")
        self.case_service = case_service

    def save_evidence(self, case_id: str, filename: str, content_type: str,
                      data: bytes, uploaded_by: str = "anonymous") -> Optional[Dict]:
        case = self.case_service.get_case(case_id)
        if not case:
            return None

        file_id = self.fs.put(
            data,
            filename=filename,
            content_type=content_type,
            case_id=case_id,
            uploaded_by=uploaded_by,
            uploaded_at=datetime.now()
        )
        self.case_service.increment_evidence_count(case_id)
        return self._serialize_meta({
            "_id": file_id,
            "filename": filename,
            "content_type": content_type,
            "case_id": case_id,
            "uploaded_by": uploaded_by,
            "length": len(data),
            "uploaded_at": datetime.now()
        })

    def list_evidence(self, case_id: str) -> List[Dict]:
        files = list(self.fs.find({"case_id": case_id}).sort("uploaded_at", -1))
        return [self._serialize_meta(f) for f in files]

    def get_evidence(self, file_id: str) -> Optional[Tuple[Dict, bytes]]:
        try:
            file_doc = self.fs.get(ObjectId(file_id))
        except Exception:
            return None
        if not file_doc:
            return None
        data = file_doc.read()
        meta = {
            "_id": str(file_doc._id),
            "filename": file_doc.filename,
            "content_type": getattr(file_doc, "content_type", "application/octet-stream"),
            "case_id": getattr(file_doc, "case_id", None),
            "uploaded_by": getattr(file_doc, "uploaded_by", None),
            "length": file_doc.length,
            "uploaded_at": getattr(file_doc, "uploaded_at", None)
        }
        return meta, data

    def delete_evidence(self, file_id: str) -> bool:
        try:
            file_doc = self.fs.get(ObjectId(file_id))
        except Exception:
            return False
        if not file_doc:
            return False
        self.fs.delete(ObjectId(file_id))
        return True

    @staticmethod
    def _serialize_meta(doc) -> Dict:
        uploaded_at = getattr(doc, "uploaded_at", None) or doc.get("uploaded_at")
        return {
            "file_id": str(doc["_id"]),
            "filename": doc.get("filename") or getattr(doc, "filename", None),
            "content_type": doc.get("content_type") or getattr(doc, "content_type", "application/octet-stream"),
            "case_id": doc.get("case_id") or getattr(doc, "case_id", None),
            "uploaded_by": doc.get("uploaded_by") or getattr(doc, "uploaded_by", None),
            "size": doc.get("length") or getattr(doc, "length", 0),
            "uploaded_at": uploaded_at.isoformat() if uploaded_at else None
        }
