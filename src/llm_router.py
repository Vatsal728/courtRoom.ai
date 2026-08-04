import os
import sys
import io
import time
import json
import requests
import httpx
from typing import Dict, AsyncIterator
from dotenv import load_dotenv

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

load_dotenv(override=True)

SYSTEM_PROMPT = """You are an elite Indian Legal AI assistant. Resolve the user's issue with high legal accuracy.

STRICT COMPLIANCE RULES:
1. THE IPC IS REPLACED. As of July 1, 2024, the Indian Penal Code (IPC) was replaced by Bharatiya Nyaya Sanhita (BNS), 2023. You MUST map all criminal sections to BNS 2023:
   - Replace IPC 420 with BNS Section 318 (Cheating).
   - Replace IPC 499/500 with BNS Section 356 (Defamation).
   - Replace IPC 503/506 with BNS Section 351 (Criminal Intimidation).
   - Replace IPC 378/379 with BNS Section 303 (Theft).
   - Replace IPC 302 with BNS Section 103 (Murder).
   - Replace CrPC with BNSS 2023 (Bharatiya Nagarik Suraksha Sanhita).
   - Replace Evidence Act with BSA 2023 (Bharatiya Sakshya Adhiniyam).
2. If retrieved context documents contain older IPC sections, state: 'Under BNS 2023 (formerly IPC Section X...)'.
3. Use ONLY the sections present in the Context above. Cite exact section numbers from the Context. If no Context section matches the query, say 'No specific provision found for this issue' instead of inventing sections. Never invent section numbers, titles, or penalties not present in the Context. The IPC->BNS mappings in rule 1 are illustrative only - you must NOT quote them unless the exact section also appears in the Context. If a section number is not in the Context, do not include it in applicable_sections.
4. You MUST return a valid JSON object. No markdown, no extra text, no code fences.

Return this exact JSON structure:
{
  "short_answer": "2-3 sentence summary",
  "is_this_illegal": "explain legality",
  "criminal_route": {
    "applicable_sections": ["BNS 2023 Section ..."],
    "penalties": ["..."],
    "procedure": ["..."]
  },
  "civil_route": {
    "remedies": ["..."],
    "compensation_range": "estimated range",
    "procedure": ["..."]
  },
  "compensation_claims": ["..."],
  "evidence_needed": ["..."],
  "practical_steps": ["step1", "step2", "step3", "step4", "step5", "step6"]
}"""

USER_PROMPT_TEMPLATE = """Context:
{context}

AVAILABLE SECTIONS — applicable_sections and all section citations MUST be chosen ONLY from this list (never invent or use the IPC->BNS mapping examples):
{available}

Query:
{query}"""

LLM_TIMEOUT_ERR = {
    "short_answer": "AI service is temporarily unavailable. Please try again in a few moments.",
    "is_this_illegal": "Unable to determine legality at this time.",
    "criminal_route": {"applicable_sections": [], "penalties": [], "procedure": []},
    "civil_route": {"remedies": [], "compensation_range": "", "procedure": []},
    "compensation_claims": [],
    "evidence_needed": [],
    "practical_steps": ["Contact a local lawyer for immediate assistance"]
}


class LLMRouter:
    """Route queries: Local Ollama -> Static error"""

    def __init__(self):
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        self.ollama_timeout = int(os.getenv("OLLAMA_TIMEOUT", "300"))
        self._session = requests.Session()

    def _retry(self, fn, max_attempts: int = 2, backoff: float = 1.0):
        last_exc = None
        for attempt in range(max_attempts):
            try:
                return fn()
            except (requests.ConnectionError, requests.Timeout) as e:
                last_exc = e
                if attempt < max_attempts - 1:
                    time.sleep(backoff * (2 ** attempt))
            except Exception as e:
                raise e
        raise last_exc

    def _call_ollama(self, prompt: str) -> str:
        response = self._session.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "temperature": 0.1,
                "options": {"num_predict": 900, "num_ctx": 8192},
                "keep_alive": "30m"
            },
            timeout=self.ollama_timeout
        )
        if response.status_code == 200:
            return response.json().get("response", "")
        raise Exception(f"Ollama error: {response.status_code}")

    def generate_response(self, context: str, query: str, available: str = "") -> str:
        """Generate a legal analysis using local Ollama, then static error."""
        user_prompt = USER_PROMPT_TEMPLATE.format(context=context, query=query, available=available)

        try:
            print("  -> Trying local Ollama...")
            prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
            return self._retry(lambda: self._call_ollama(prompt))
        except Exception as e:
            print(f"  \u26a0\ufe0f Ollama unavailable: {e}")
            print("  \u21aa  Using static error response...")
            return str(LLM_TIMEOUT_ERR)

    async def stream_generate(self, context: str, query: str, available: str = "") -> AsyncIterator[str]:
        """Stream incremental JSON text chunks from Ollama (async).

        Yields raw LLM output strings so callers can forward them as SSE.
        Cancellation: when the caller stops iterating (client disconnect),
        the httpx response is closed and generation is aborted server-side.
        """
        user_prompt = USER_PROMPT_TEMPLATE.format(context=context, query=query, available=available)
        prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
        url = f"{self.ollama_url}/api/generate"
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": True,
            "format": "json",
            "temperature": 0.1,
            "options": {"num_predict": 900, "num_ctx": 8192},
            "keep_alive": "30m"
        }
        async with httpx.AsyncClient(timeout=self.ollama_timeout) as client:
            async with client.stream("POST", url, json=payload) as resp:
                if resp.status_code != 200:
                    raise Exception(f"Ollama error: {resp.status_code}")
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    piece = data.get("response", "")
                    if piece:
                        yield piece
                    if data.get("done", False):
                        break


if __name__ == "__main__":
    router = LLMRouter()

    test_context = "Consumer Protection Act 2019 Section 35: Consumer can file complaint for defective products within 2 years"
    test_query = "I bought a defective phone. What can I do?"

    response = router.generate_response(test_context, test_query)
    print(f"Response: {response}")
