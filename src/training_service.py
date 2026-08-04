"""training_service.py - Fine-tuning dataset export (Phase 19)
Builds prompt/completion JSONL from user queries and feedback.
"""
import os
import json
from typing import Optional

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv(override=True)

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "courtroom_ai")


class TrainingService:
    def __init__(self):
        self.db = MongoClient(MONGODB_URI)[MONGODB_DB]

    def export_dataset(self, output_path: str = "training_data.jsonl",
                       min_confidence: float = 0.5) -> dict:
        """Export training data from queries collection + chat sessions."""
        query_docs = list(
            self.db.queries.find(
                {"confidence": {"$gte": min_confidence}}
            ).sort("created_at", -1)
        )

        lines = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for doc in query_docs:
                prompt = doc.get("query")
                response = doc.get("response") or doc.get("short_answer")
                if not prompt or not response:
                    continue
                completion = response
                if isinstance(completion, dict):
                    completion = json.dumps(completion, ensure_ascii=False)
                f.write(json.dumps(
                    {"prompt": prompt, "completion": str(completion)},
                    ensure_ascii=False
                ) + "\n")
                lines += 1

        return {
            "output_file": output_path,
            "examples_exported": lines,
            "min_confidence": min_confidence,
        }