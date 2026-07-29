import os
import sys
import io
from typing import Dict
from dotenv import load_dotenv

# Set console output encoding to UTF-8 to prevent UnicodeEncodeError on Windows terminals
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

load_dotenv()

class LLMRouter:
    """Route queries: Gemini Primary → Ollama Fallback"""
    
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "gemini")
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "qwen3:4b")
        self.ollama_timeout = int(os.getenv("OLLAMA_TIMEOUT", "300"))
    
    def get_gemini_response(self, context: str, query: str) -> str:
        """Call Gemini API - Primary"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_key)
            
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            prompt = f"""You are an elite Indian Legal AI assistant. Resolve the user's issue with high legal accuracy and clean formatting.

STRICT COMPLIANCE RULES:
1. NEVER leave blank headers or empty asterisks like '**, Section...'. Ensure every single bullet point has complete, non-truncated text.
2. THE IPC IS REPLACED. As of July 1, 2024, the Indian Penal Code (IPC) was replaced by Bharatiya Nyaya Sanhita (BNS), 2023. You MUST map all criminal sections to BNS 2023:
   - Replace IPC 420 with BNS Section 318 (Cheating).
   - Replace IPC 499/500 with BNS Section 356 (Defamation).
   - Replace IPC 503/506 with BNS Section 351 (Criminal Intimidation).
   - Replace IPC 378/379 with BNS Section 303 (Theft).
   - Replace IPC 302 with BNS Section 103 (Murder).
   - Replace CrPC with BNSS 2023 (Bharatiya Nagarik Suraksha Sanhita).
   - Replace Evidence Act with BSA 2023 (Bharatiya Sakshya Adhiniyam).
3. If retrieved context documents contain older IPC sections, state: 'Under BNS 2023 (formerly IPC Section X)...'.

You MUST structure your response exactly using these headers:
### SHORT ANSWER
[Provide a concise 2-3 sentence summary answer]

### IS THIS ILLEGAL?
[Explain if the action is illegal and cite BNS 2023 and relevant special laws]

### CRIMINAL ROUTE
- Sections: [List applicable BNS 2023 sections]
- Penalties: [List penalties/imprisonment terms]
- Procedure: [Step-by-step BNSS 2023 criminal process]

### CIVIL ROUTE
- Remedies: [List civil remedies/injunctions]
- Compensation Range: [Estimated compensation or damages]
- Procedure: [Step-by-step civil filing process]

### COMPENSATION CLAIMS
- [List specific claim 1]
- [List specific claim 2]

### EVIDENCE NEEDED
- [List evidence item 1]
- [List evidence item 2]

### PRACTICAL STEPS
1. [Step 1]
2. [Step 2]
3. [Step 3]
4. [Step 4]
5. [Step 5]
6. [Step 6]

Context (relevant law sections):
{context}

User Query:
{query}
"""
            
            response = model.generate_content(prompt, timeout=30)
            return response.text
        
        except Exception as e:
            print(f"⚠️  Gemini unavailable: {type(e).__name__}")
            return self.get_ollama_response(context, query)
    
    def get_ollama_response(self, context: str, query: str) -> str:
        """Call local Ollama (Qwen3:4b) - Fallback"""
        try:
            import requests
            
            prompt = f"""You are an elite Indian Legal AI assistant. Resolve the user's issue with high legal accuracy and clean formatting.

STRICT COMPLIANCE RULES:
1. NEVER leave blank headers or empty asterisks like '**, Section...'. Ensure every single bullet point has complete, non-truncated text.
2. THE IPC IS REPLACED. As of July 1, 2024, the Indian Penal Code (IPC) was replaced by Bharatiya Nyaya Sanhita (BNS), 2023. You MUST map all criminal sections to BNS 2023:
   - Replace IPC 420 with BNS Section 318 (Cheating).
   - Replace IPC 499/500 with BNS Section 356 (Defamation).
   - Replace IPC 503/506 with BNS Section 351 (Criminal Intimidation).
   - Replace IPC 378/379 with BNS Section 303 (Theft).
   - Replace IPC 302 with BNS Section 103 (Murder).
   - Replace CrPC with BNSS 2023 (Bharatiya Nagarik Suraksha Sanhita).
   - Replace Evidence Act with BSA 2023 (Bharatiya Sakshya Adhiniyam).
3. If retrieved context documents contain older IPC sections, state: 'Under BNS 2023 (formerly IPC Section X)...'.

You MUST structure your response exactly using these headers:
### SHORT ANSWER
[Provide a concise 2-3 sentence summary answer]

### IS THIS ILLEGAL?
[Explain if the action is illegal and cite BNS 2023 and relevant special laws]

### CRIMINAL ROUTE
- Sections: [List applicable BNS 2023 sections]
- Penalties: [List penalties/imprisonment terms]
- Procedure: [Step-by-step BNSS 2023 criminal process]

### CIVIL ROUTE
- Remedies: [List civil remedies/injunctions]
- Compensation Range: [Estimated compensation or damages]
- Procedure: [Step-by-step civil filing process]

### COMPENSATION CLAIMS
- [List specific claim 1]
- [List specific claim 2]

### EVIDENCE NEEDED
- [List evidence item 1]
- [List evidence item 2]

### PRACTICAL STEPS
1. [Step 1]
2. [Step 2]
3. [Step 3]
4. [Step 4]
5. [Step 5]
6. [Step 6]

Context:
{context}

Query: {query}
"""
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.1
                },
                timeout=self.ollama_timeout
            )
            
            if response.status_code == 200:
                result = response.json().get("response", "No response from model")
                return result
            else:
                return f"⚠️  Ollama error: {response.status_code}"
        
        except requests.exceptions.ConnectionError:
            return "❌ Ollama not running. Start Ollama before using courtRoom.ai"
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def generate_response(self, context: str, query: str) -> str:
        """Route to Gemini, fallback to Ollama"""
        if self.provider.lower() == "ollama":
            # Force Ollama mode
            return self.get_ollama_response(context, query)
        else:
            # Default: try Gemini first
            return self.get_gemini_response(context, query)

if __name__ == "__main__":
    router = LLMRouter()
    
    test_context = "Consumer Protection Act 2019 Section 35: Consumer can file complaint for defective products within 2 years"
    test_query = "I bought a defective phone. What can I do?"
    
    response = router.generate_response(test_context, test_query)
    print(f"Response: {response}")
