import os
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

os.environ["CHROMA_TELEMETRY"] = "false"
os.environ["CHROMA_ANONYMIZED_TELEMETRY"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"

import sys
import io
from pathlib import Path

# Set console output encoding to UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag_pipeline import RAGPipeline as ImprovedRAGPipeline
from src.llm_router import LLMRouter
from src.classifier import CaseTypeClassifier
from src.nlp_pipeline import NLPPipeline
from src.domain_classifier import DomainClassifier
from src.response_formatter import ResponseFormatter, format_legal_response
from typing import Dict, Any

class FullRAGSystem:
    """Complete RAG pipeline: NLP → Classify → Retrieve → Generate"""
    
    def __init__(self):
        """Initialize improved RAG system"""
        self.nlp = NLPPipeline()
        self.classifier = CaseTypeClassifier()
        
        # Add new components
        self.domain_classifier = DomainClassifier()
        self.response_formatter = ResponseFormatter()
        self.improved_rag = ImprovedRAGPipeline()  # Use improved pipeline
        self.llm_router = LLMRouter()
        
        # Disable telemetry
        os.environ["CHROMA_TELEMETRY"] = "false"
        os.environ["CHROMA_ANONYMIZED_TELEMETRY"] = "false"
        
    def build_knowledge_base(self):
        """Build the improved RAG knowledge base"""
        return self.improved_rag.build_knowledge_base()
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process query with improved classification and formatting
        
        Returns structured response like NyayGuru
        """
        
        try:
            # 1. Classify domain
            print("1️⃣  Classifying domain...")
            domain, domain_confidence, secondary = self.domain_classifier.classify(query)
            response_type = self.domain_classifier.get_response_type(domain, secondary)
            
            # 2. Retrieve with improved RAG
            print("2️⃣  Retrieving relevant laws...")
            sources = self.improved_rag.retrieve_with_metadata(query, top_k=5)
            
            # 3. Get LLM response (pass context and query)
            print("3️⃣  Generating legal analysis...")
            context = "\n".join([s["content"] for s in sources])
            llm_response = self.llm_router.generate_response(context, query)
            
            # 4. Format response like NyayGuru
            print("4️⃣  Formatting response (NyayGuru-style)...")
            formatted_response = format_legal_response(
                query=query,
                llm_response=llm_response,
                sources=sources,
                domain=domain,
                confidence=domain_confidence
            )
            
            print("5️⃣  Adding metadata...")
            # Compatibility keys for frontend UI rendering
            if not formatted_response.get("response"):
                formatted_response["response"] = formatted_response.get("full_response") or formatted_response.get("short_answer")
            formatted_response["domain"] = formatted_response.get("response_type")
            formatted_response["confidence"] = formatted_response.get("confidence_score")
            formatted_response["stored"] = True
            
            print("✅ Response complete!")
            return formatted_response
            
        except Exception as e:
            return {
                "error": str(e),
                "status": "failed"
            }

if __name__ == "__main__":
    system = FullRAGSystem()
    
    test_queries = [
        "Someone hacked my website, stole my company's database, and published it online.",
        "A person assaulted me outside the market and stole my wallet.",
        "The landlord is trying to evict me."
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        result = system.process_query(query)
        print(f"Result: {result}")
