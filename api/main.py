import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime

from src.full_rag import FullRAGSystem
from src.inference_engine import InferenceEngine
from src.nlp_pipeline import NLPPipeline
from src.classifier import CaseTypeClassifier
from src.agents.notice_agent import LegalNoticeAgent
from src.agents.evidence_agent import EvidenceChecklistAgent
from src.agents.rti_agent import RTIApplicationAgent
from src.agents.deadline_agent import DeadlineTrackerAgent
from api.auth import router as auth_router

load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "courtroom_ai")
db = MongoClient(MONGODB_URI)[MONGODB_DB]

app = FastAPI(title="courtRoom.ai API", version="1.0")
app.include_router(auth_router)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize systems
try:
    rag_system = FullRAGSystem()
    print("✓ Connected to local RAG knowledge base")
except Exception as e:
    print(f"⚠️  RAG system initialization failed: {e}")
    rag_system = None

inference = InferenceEngine()
nlp = NLPPipeline()
classifier = CaseTypeClassifier()
notice_agent = LegalNoticeAgent()
evidence_agent = EvidenceChecklistAgent()
rti_agent = RTIApplicationAgent()
deadline_agent = DeadlineTrackerAgent()

class QueryStorage:
    def __init__(self):
        self.db = MongoClient(MONGODB_URI)[MONGODB_DB]
    
    def store_query(self, user_id: str, query_data: Dict) -> str:
        """Store query and results in MongoDB"""
        document = {
            "user_id": user_id,
            "query": query_data["query"],
            "domain": query_data["domain"],
            "confidence": query_data["confidence"],
            "response": query_data["response"],
            "sources": query_data["sources"],
            "created_at": datetime.now(),
            "status": "completed"
        }
        result = self.db.queries.insert_one(document)
        return str(result.inserted_id)
    
    def store_pdf_notice(self, user_id: str, case_id: str, pdf_data: Dict) -> str:
        """Store generated PDF metadata"""
        document = {
            "user_id": user_id,
            "case_id": case_id,
            "filename": pdf_data["filename"],
            "pdf_path": pdf_data["pdf_path"],
            "sender_name": pdf_data["sender_name"],
            "recipient_name": pdf_data["recipient_name"],
            "issue_type": pdf_data["issue_type"],
            "demand_amount": pdf_data["demand_amount"],
            "created_at": datetime.now(),
            "downloaded": False,
            "download_count": 0
        }
        result = self.db.pdfs.insert_one(document)
        return str(result.inserted_id)
    
    def get_user_queries(self, user_id: str, limit: int = 50):
        """Retrieve user's query history"""
        queries = list(self.db.queries.find(
            {"user_id": user_id}
        ).sort("created_at", -1).limit(limit))
        for q in queries:
            q["_id"] = str(q["_id"])
        return queries
    
    def get_user_pdfs(self, user_id: str):
        """Retrieve user's generated PDFs"""
        pdfs = list(self.db.pdfs.find(
            {"user_id": user_id}
        ).sort("created_at", -1))
        for p in pdfs:
            p["_id"] = str(p["_id"])
        return pdfs

storage = QueryStorage()

# Request models
class QueryRequest(BaseModel):
    query: str
    language: str = "en"

class NoticeRequest(BaseModel):
    sender_name: str
    sender_address: str
    recipient_name: str
    recipient_address: str
    issue_type: str
    issue_description: str
    applicable_section: str
    demand_amount: str

class DeadlineRequest(BaseModel):
    case_type: str
    incident_date: str  # DD-MM-YYYY format

# Endpoints
@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "courtRoom.ai", "version": "1.0"}

@app.post("/query")
def process_query(req: QueryRequest, user_id: str = "anonymous"):
    """Process legal query with improved system"""
    try:
        if rag_system is None:
            raise Exception("RAG system not initialized")
        
        # Use improved process_query
        result = rag_system.process_query(req.query)
        
        # Compatibility keys for frontend UI rendering
        if not result.get("response") or not str(result.get("response")).strip():
            result["response"] = result.get("full_response") or result.get("short_answer") or "No output could be generated for this query."
        result["domain"] = result.get("response_type")
        result["confidence"] = result.get("confidence_score")
        
        # Store in MongoDB
        query_id = storage.store_query(user_id, {
            "query": req.query,
            "domain": result.get("response_type"),
            "confidence": result.get("confidence_score"),
            "response": result.get("response"),
            "full_response": result,
            "sources": result.get("sources", [])
        })
        
        result["query_id"] = query_id
        result["stored"] = True
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-notice")
def generate_legal_notice(req: NoticeRequest, user_id: str = "anonymous") -> Dict:
    """Generate notice and store metadata in MongoDB"""
    try:
        case_data = req.dict()
        
        # Generate filename with user's name
        filename = f"Legal_Notice_{case_data['sender_name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        case_data['filename'] = filename
        
        pdf_path = notice_agent.generate_notice(case_data, output_path=f"output/{filename}")
        
        # Store metadata in MongoDB
        pdf_id = storage.store_pdf_notice(user_id, "case_001", {
            "filename": filename,
            "pdf_path": pdf_path,
            "sender_name": case_data["sender_name"],
            "recipient_name": case_data["recipient_name"],
            "issue_type": case_data["issue_type"],
            "demand_amount": case_data["demand_amount"]
        })
        
        return {
            "status": "success",
            "pdf_id": pdf_id,
            "pdf_path": pdf_path,
            "filename": filename,
            "message": "Legal notice generated and saved"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/{user_id}/queries")
def get_query_history(user_id: str):
    """Get user's query history"""
    return storage.get_user_queries(user_id)

@app.get("/user/{user_id}/pdfs")
def get_pdf_history(user_id: str):
    """Get user's generated PDFs"""
    return storage.get_user_pdfs(user_id)

@app.get("/pdf/{pdf_id}/download")
def download_pdf(pdf_id: str):
    """Download PDF file"""
    pdf_doc = db.pdfs.find_one({"_id": ObjectId(pdf_id)})
    if not pdf_doc:
        raise HTTPException(status_code=404, detail="PDF not found")
    
    # Update download count
    db.pdfs.update_one(
        {"_id": ObjectId(pdf_id)},
        {"$inc": {"download_count": 1}, "$set": {"downloaded": True}}
    )
    
    # Return file
    return FileResponse(
        path=pdf_doc["pdf_path"],
        filename=pdf_doc["filename"],
        media_type="application/pdf"
    )

@app.get("/evidence/{domain}")
def get_evidence_checklist(domain: str) -> Dict:
    """Get evidence checklist for domain"""
    try:
        checklist = evidence_agent.get_checklist(domain)
        return checklist
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rti-application")
def generate_rti_app(data: Dict) -> Dict:
    """Generate RTI application"""
    try:
        rti_app = rti_agent.generate_rti_application(data)
        return {
            "status": "success",
            "application": rti_app
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/deadline")
def check_deadline(req: DeadlineRequest) -> Dict:
    """Check filing deadline"""
    try:
        result = deadline_agent.calculate_deadline(req.case_type, req.incident_date)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
