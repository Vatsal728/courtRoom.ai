from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv(override=True)

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DB")

client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB]

# Create collections
collections = {
    "cases": {
        "indexes": ["user_id", "case_type", "created_at"]
    },
    "chunks": {
        "indexes": ["domain", "source", "chunk_id"]
    },
    "sessions": {
        "indexes": ["session_id", "user_id", "timestamp"]
    },
    "training_data": {
        "indexes": ["domain", "label"]
    },
    "evidence": {
        "indexes": ["case_id", "uploaded_at"]
    }
}

for collection_name, config in collections.items():
    if collection_name not in db.list_collection_names():
        db.create_collection(collection_name)
        print(f"Created collection: {collection_name}")
        
        # Create indexes
        collection = db[collection_name]
        for index_field in config["indexes"]:
            collection.create_index(index_field)
        print(f"   Indexes: {config['indexes']}")
    else:
        print(f"Collection {collection_name} already exists")

print("\nDatabase initialization complete")
print(f"Database: {MONGODB_DB}")
print(f"Collections: {db.list_collection_names()}")
