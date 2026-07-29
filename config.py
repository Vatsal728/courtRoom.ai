"""
config.py - Configuration for courtRoom.ai
"""

import os
from dotenv import load_dotenv

load_dotenv()

# RAG Configuration
RAG_CONFIG = {
    "pdf_directory": "data/pdfs",
    "chroma_db_path": "chroma_db",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "top_k_retrieval": 5,
    "bm25_weight": 0.4,
    "vector_weight": 0.6
}

# LLM Configuration
LLM_CONFIG = {
    "gemini_api_key": os.getenv("GEMINI_API_KEY"),
    "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    "ollama_model": os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
    "ollama_embed_model": "nomic-embed-text",
    "request_timeout": int(os.getenv("REQUEST_TIMEOUT", "300")),
    "temperature": 0.7,
    "max_tokens": 2048
}

# MongoDB Configuration
MONGODB_CONFIG = {
    "uri": os.getenv("MONGODB_URI", "mongodb://localhost:27017"),
    "db_name": os.getenv("MONGODB_DB", "courtroom_ai"),
    "collections": {
        "queries": "queries",
        "pdfs": "pdfs",
        "users": "users",
        "cases": "cases",
        "consultations": "consultations"
    }
}

# Classifier Configuration
CLASSIFIER_CONFIG = {
    "min_confidence": 0.3,
    "high_confidence_threshold": 0.8,
    "enable_secondary_detection": True
}

# Response Formatting
FORMATTER_CONFIG = {
    "include_short_answer": True,
    "include_criminal_route": True,
    "include_civil_route": True,
    "include_practical_steps": True,
    "include_compensation": True,
    "max_sources": 5,
    "max_remedies": 4
}

# ChromaDB Configuration
CHROMA_CONFIG = {
    "telemetry": False,
    "anonymized_telemetry": False,
    "persistent": True,
    "path": "chroma_db"
}
