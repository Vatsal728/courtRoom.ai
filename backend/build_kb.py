#!/usr/bin/env python
"""Build knowledge base from PDFs"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.full_rag import FullRAGSystem

if __name__ == "__main__":
    rag = FullRAGSystem()
    rag.build_knowledge_base()
    print("\n✅ Knowledge base built successfully!")
