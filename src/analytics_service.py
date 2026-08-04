"""analytics_service.py - Usage analytics (Phase 13)
Aggregates query, case and PDF activity from MongoDB.
"""
import os
from datetime import datetime, timedelta

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv(override=True)

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "courtroom_ai")


class AnalyticsService:
    def __init__(self):
        self.db = MongoClient(MONGODB_URI)[MONGODB_DB]

    def overview(self, days: int = 30) -> dict:
        since = datetime.now() - timedelta(days=days)

        total_queries = self.db.queries.count_documents({})
        total_cases = self.db.cases.count_documents({})
        total_pdfs = self.db.pdfs.count_documents({})

        domain_distribution = list(
            self.db.queries.aggregate([
                {"$match": {"created_at": {"$gte": since}}},
                {"$group": {"_id": "$domain", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
            ])
        )

        top_users = list(
            self.db.queries.aggregate([
                {"$group": {"_id": "$user_id", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 5},
            ])
        )

        daily_trend = list(
            self.db.queries.aggregate([
                {"$match": {"created_at": {"$gte": since}}},
                {"$group": {
                    "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}},
                    "count": {"$sum": 1},
                }},
                {"$sort": {"_id": 1}},
            ])
        )

        top_sections = list(
            self.db.queries.aggregate([
                {"$match": {"sources": {"$exists": True, "$ne": []}}},
                {"$unwind": "$sources"},
                {"$group": {
                    "_id": {
                        "act": "$sources.source_act",
                        "section": "$sources.section_number",
                    },
                    "count": {"$sum": 1},
                }},
                {"$sort": {"count": -1}},
                {"$limit": 10},
            ])
        )

        return {
            "period_days": days,
            "total_queries": total_queries,
            "total_cases": total_cases,
            "total_pdfs": total_pdfs,
            "domain_distribution": [
                {"domain": d["_id"] or "unknown", "count": d["count"]}
                for d in domain_distribution
            ],
            "top_users": [
                {"user_id": u["_id"] or "anonymous", "count": u["count"]}
                for u in top_users
            ],
            "daily_trend": [
                {"date": d["_id"], "count": d["count"]} for d in daily_trend
            ],
            "top_cited_sections": [
                {
                    "section": f"{s['_id'].get('section')}of {s['_id'].get('source')}"
                    if s["_id"] else "unknown",
                    "count": s["count"],
                }
                for s in top_sections
            ],
        }