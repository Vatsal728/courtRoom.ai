import os
import sys
import io
import time
import json
import logging
import httpx
from typing import Dict, AsyncIterator
from dotenv import load_dotenv

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

load_dotenv(override=True)

logger = logging.getLogger("courtroom-llm-router")

SYSTEM_PROMPT = """You are an elite Indian Legal AI assistant. Resolve the user's issue with high legal accuracy.

STRICT COMPLIANCE RULES:
1. CRIMINAL LAW IS REPLACED. As of July 1, 2024, the Indian Penal Code (IPC) was replaced by Bharatiya Nyaya Sanhita (BNS), 2023. Map criminal sections to BNS 2023:
   - Replace IPC 420 with BNS Section 318 (Cheating).
   - Replace IPC 499/500 with BNS Section 356 (Defamation).
   - Replace IPC 503/506 with BNS Section 351 (Criminal Intimidation).
   - Replace IPC 378/379 with BNS Section 303 (Theft).
   - Replace IPC 302 with BNS Section 103 (Murder).
   - Replace CrPC with BNSS 2023 (Bharatiya Nagarik Suraksha Sanhita).
   - Replace Evidence Act with BSA 2023 (Bharatiya Sakshya Adhiniyam).
2. LABOUR LAW IS CODIFIED. The Code on Wages 2019 (in force 21 Nov 2025) repealed the Payment of Wages Act 1936 and the Minimum Wages Act 1948. The Code on Social Security 2020 and the OSHWC Code 2020 are also in force from 21 Nov 2025; the Industrial Relations Code 2020 takes effect 21 Nov 2026 (Industrial Disputes Act 1947 remains law until then). Distinguish UNPAID CONTRACTUAL SALARY (a wage-payment claim; the amount owed is the contractually agreed wage) from STATUTORY MINIMUM WAGES (fixed under the Code on Wages' schedules; an employer must always pay at least the notified minimum wage). A worker paid above minimum wage still has a wage-payment remedy for unpaid salary. Forum: wage claims under the Code on Wages are adjudicated by the Claims Authority / Appellate Authority (Labour Commissioner) and by criminal prosecution for violations; industrial disputes (lay-off, retrenchment, dismissal) go to the Labour Court / Industrial Tribunal. If the query omits essential facts (worker's State, establishment type/sector, worker category, monthly wage, number of workers employed), state which missing facts would change the answer and give the answer under clearly-stated assumptions.
3. HISTORICAL SOURCES. If the Context contains a historical/repealed act (e.g., Payment of Wages Act 1936, Minimum Wages Act 1948, IPC, CrPC, Indian Evidence Act, Industrial Disputes Act 1947), cite it only as background: label it '(historical)' and, where the current replacement (BNS 2023 / BNSS 2023 / BSA 2023 / Code on Wages 2019 / Industrial Relations Code 2020) is also in the Context, cite the replacement as the operative law.
4. Use ONLY the sections present in the Context above. Cite exact section numbers from the Context. If no Context section matches the query, say 'No specific provision found for this issue' instead of inventing sections. Never invent section numbers, titles, or penalties not present in the Context. The IPC->BNS mappings in rule 1 are illustrative only - you must NOT quote them unless the exact section also appears in the Context. If a section number is not in the Context, do not include it in applicable_sections.
5. LEGAL CITATION RULES: every legal conclusion must be traceable to a source in the Context; never cite a provision merely because its title contains a similar keyword; if the Context does not establish whether conduct is illegal, say the evidence is insufficient instead of guessing; never infer penalties unless the Context states them; never invent compensation amounts.
6. PENALTIES MUST BE PER SECTION. In `criminal_route.penalties`, key each penalty by its exact section (e.g. "Bharatiya Nyaya Sanhita 2023 Section 303") and give only that section's punishment as stated in the Context. Never merge punishments from different sections into one entry.
7. You MUST return a valid JSON object. No markdown, no extra text, no code fences.

Return this exact JSON structure:
{
  "short_answer": "2-3 sentence summary",
  "is_this_illegal": "explain legality",
  "criminal_route": {
    "applicable_sections": ["BNS 2023 Section ..."],
    "penalties": {"BNS 2023 Section 303": "exact punishment from the context"},
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

# Used only for small talk / chat-openers (greetings, thanks, casual chat).
# The LLM replies naturally WITHOUT RAG context; when the user starts stating
# a legal problem, the full RAG pipeline takes over.
CHAT_SYSTEM_PROMPT = """You are the friendly conversational mode of an Indian legal assistant. The user is making small talk (a greeting, thanks, or casual chat) rather than describing a legal problem yet.

Rules:
1. Reply briefly and naturally, like a helpful assistant would in a chat. 1-3 sentences.
2. Do NOT invent, name, or cite any law, act, or section. Do not give legal advice.
3. Gently invite the user to describe what happened (who, what, when) so the assistant can switch to full legal analysis.
4. If they asked a simple casual question, answer it directly and briefly.
5. Keep it warm and plain-English; never use markdown."""

CHAT_FALLBACK_REPLY = (
    "Hi! I'm your Indian legal assistant. Tell me what happened — with whom, "
    "and when — and I'll help you understand your legal options."
)


class LLMRouter:
    """Route queries: Groq -> Static error.

    Groq (llama-3.3-70b-versatile) is the engine for speed and complete JSON;
    a static error object is the fallback when Groq is down/rate-limited.
    """

    _GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_model = os.getenv("GROQ_GENERATION_MODEL", "llama-3.3-70b-versatile")
        self.groq_max_tokens = int(os.getenv("GROQ_GENERATION_MAX_TOKENS", "2048"))
        self.groq_timeout = float(os.getenv("GROQ_GENERATION_TIMEOUT", "60"))
        self.groq_ssl_verify = os.getenv("GROQ_SSL_VERIFY", "1") == "1"
        self.groq_enabled = os.getenv("GROQ_ENABLED", "1") == "1" and bool(self.groq_api_key)
        self._down_until = 0.0
        self.last_was_error = False

        # Local Ollama fallback engine (used when Groq is down/disabled so the
        # answer is still generated from the retrieved context, never static).
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        self.ollama_model = os.getenv("OLLAMA_GENERATION_MODEL", "qwen2.5:3b")
        self.ollama_timeout = float(os.getenv("OLLAMA_GENERATION_TIMEOUT", "180"))
        self.ollama_enabled = os.getenv("OLLAMA_GENERATION_ENABLED", "1") == "1"

        # Which engine produced the last response: "groq" | "ollama" | "static".
        self.generation_engine = "static"

    @property
    def _groq_down(self) -> bool:
        return time.monotonic() < self._down_until

    def _mark_groq_down(self, seconds: float = 30.0):
        self._down_until = time.monotonic() + seconds
        logger.warning("Groq generation marked down for %ss", int(seconds))

    @staticmethod
    def _groq_retryable(err) -> bool:
        msg = str(err).lower()
        return any(h in msg for h in (
            "429", "rate limit", "403", "blocked", "forbidden", "unauthorized", "401",
        ))

    def _call_groq(self, prompt: str) -> str:
        if not self.groq_enabled or self._groq_down:
            raise RuntimeError("Groq unavailable")
        from groq import Groq
        client = Groq(api_key=self.groq_api_key, max_retries=1)
        resp = client.chat.completions.create(
            model=self.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=self.groq_max_tokens,
            response_format={"type": "json_object"},
            timeout=self.groq_timeout,
        )
        content = str(resp.choices[0].message.content or "").strip()
        if not content:
            raise RuntimeError("Groq empty response")
        return content

    def _call_ollama(self, prompt: str) -> str:
        """Generate via a local Ollama model (qwen2.5:3b by default).

        Same system prompt + JSON contract as Groq, so downstream formatting
        treats both engines identically. `format: "json"` requests strict JSON.
        """
        import httpx
        payload = {
            "model": self.ollama_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1},
        }
        with httpx.Client(timeout=self.ollama_timeout, verify=False) as client:
            resp = client.post(f"{self.ollama_base_url}/api/chat", json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"Ollama error: {resp.status_code}")
            content = str(resp.json().get("message", {}).get("content", "")).strip()
            if not content:
                raise RuntimeError("Ollama empty response")
            return content

    def generate_response(self, context: str, query: str, available: str = "") -> str:
        """Generate a legal analysis: Groq -> Ollama (local) -> static error."""
        user_prompt = USER_PROMPT_TEMPLATE.format(context=context, query=query, available=available)
        prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"

        if self.groq_enabled and not self._groq_down:
            try:
                print("  -> Trying Groq generation...")
                resp = self._call_groq(prompt)
                self.last_was_error = False
                self.generation_engine = "groq"
                return resp
            except Exception as e:
                print(f"  \u26a0\ufe0f Groq generation unavailable: {e}")
                if self._groq_retryable(e):
                    self._mark_groq_down()

        if self.ollama_enabled:
            try:
                print(f"  -> Trying local Ollama ({self.ollama_model}) generation...")
                resp = self._call_ollama(prompt)
                self.last_was_error = False
                self.generation_engine = "ollama"
                return resp
            except Exception as e:
                print(f"  \u26a0\ufe0f Ollama generation unavailable: {e}")

        print("  \u21aa  Using static error response...")
        self.last_was_error = True
        self.generation_engine = "static"
        return str(LLM_TIMEOUT_ERR)

    async def _stream_groq(self, prompt: str) -> AsyncIterator[str]:
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.groq_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": self.groq_max_tokens,
            "response_format": {"type": "json_object"},
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=self.groq_timeout, verify=self.groq_ssl_verify) as client:
            async with client.stream("POST", self._GROQ_URL, json=payload, headers=headers) as resp:
                if resp.status_code != 200:
                    raise Exception(f"Groq error: {resp.status_code}")
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                    piece = delta.get("content") or ""
                    if piece:
                        yield piece

    async def _stream_ollama(self, prompt: str) -> AsyncIterator[str]:
        """Stream tokens from a local Ollama model (NDJSON over /api/chat)."""
        import httpx
        payload = {
            "model": self.ollama_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "format": "json",
            "options": {"temperature": 0.1},
        }
        async with httpx.AsyncClient(timeout=self.ollama_timeout, verify=False) as client:
            async with client.stream("POST", f"{self.ollama_base_url}/api/chat", json=payload) as resp:
                if resp.status_code != 200:
                    raise Exception(f"Ollama error: {resp.status_code}")
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    piece = (chunk.get("message") or {}).get("content") or ""
                    if piece:
                        yield piece
                    if chunk.get("done"):
                        break

    # ── Conversational chat (no RAG) ─────────────────────────────────
    def _call_groq_chat(self, prompt: str) -> str:
        if not self.groq_enabled or self._groq_down:
            raise RuntimeError("Groq unavailable")
        from groq import Groq
        client = Groq(api_key=self.groq_api_key, max_retries=1)
        resp = client.chat.completions.create(
            model=self.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=200,
            timeout=self.groq_timeout,
        )
        content = str(resp.choices[0].message.content or "").strip()
        if not content:
            raise RuntimeError("Groq empty response")
        return content

    def _call_ollama_chat(self, prompt: str) -> str:
        import httpx
        payload = {
            "model": self.ollama_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.6},
        }
        with httpx.Client(timeout=self.ollama_timeout, verify=False) as client:
            resp = client.post(f"{self.ollama_base_url}/api/chat", json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"Ollama error: {resp.status_code}")
            content = str(resp.json().get("message", {}).get("content", "")).strip()
            if not content:
                raise RuntimeError("Ollama empty response")
            return content

    def generate_chat_reply(self, query: str) -> str:
        """Small-talk reply WITHOUT RAG context. Groq -> local Ollama."""
        prompt = f"{CHAT_SYSTEM_PROMPT}\n\nUser: {query}\nAssistant:"
        if self.groq_enabled and not self._groq_down:
            try:
                resp = self._call_groq_chat(prompt)
                self.generation_engine = "groq"
                return resp
            except Exception as e:
                print(f"  \u26a0\ufe0f Groq chat unavailable: {e}")
                if self._groq_retryable(e):
                    self._mark_groq_down()
        if self.ollama_enabled:
            try:
                resp = self._call_ollama_chat(prompt)
                self.generation_engine = "ollama"
                return resp
            except Exception as e:
                print(f"  \u26a0\ufe0f Ollama chat unavailable: {e}")
        raise RuntimeError("No chat engine available")

    async def _stream_groq_chat(self, prompt: str) -> AsyncIterator[str]:
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.groq_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.6,
            "max_tokens": 200,
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=self.groq_timeout, verify=self.groq_ssl_verify) as client:
            async with client.stream("POST", self._GROQ_URL, json=payload, headers=headers) as resp:
                if resp.status_code != 200:
                    raise Exception(f"Groq error: {resp.status_code}")
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                    piece = delta.get("content") or ""
                    if piece:
                        yield piece

    async def _stream_ollama_chat(self, prompt: str) -> AsyncIterator[str]:
        import httpx
        payload = {
            "model": self.ollama_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "options": {"temperature": 0.6},
        }
        async with httpx.AsyncClient(timeout=self.ollama_timeout, verify=False) as client:
            async with client.stream("POST", f"{self.ollama_base_url}/api/chat", json=payload) as resp:
                if resp.status_code != 200:
                    raise Exception(f"Ollama error: {resp.status_code}")
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    piece = (chunk.get("message") or {}).get("content") or ""
                    if piece:
                        yield piece
                    if chunk.get("done"):
                        break

    async def stream_chat_reply(self, query: str) -> AsyncIterator[str]:
        """Stream a small-talk reply WITHOUT RAG context (Groq -> Ollama)."""
        prompt = f"{CHAT_SYSTEM_PROMPT}\n\nUser: {query}\nAssistant:"
        streamed_any = False
        if self.groq_enabled and not self._groq_down:
            try:
                async for piece in self._stream_groq_chat(prompt):
                    streamed_any = True
                    yield piece
                if streamed_any:
                    self.generation_engine = "groq"
                    return
            except Exception as e:
                print(f"  \u26a0\ufe0f Groq chat stream unavailable: {e}")
                if self._groq_retryable(e):
                    self._mark_groq_down()
        if self.ollama_enabled:
            try:
                async for piece in self._stream_ollama_chat(prompt):
                    streamed_any = True
                    yield piece
                if streamed_any:
                    self.generation_engine = "ollama"
                    return
            except Exception as e:
                print(f"  \u26a0\ufe0f Ollama chat stream unavailable: {e}")
        if not streamed_any:
            raise RuntimeError("No chat engine available")

    async def stream_generate(self, context: str, query: str, available: str = "") -> AsyncIterator[str]:
        """Stream incremental JSON text chunks (async).

        Engines: Groq (fast) -> local Ollama fallback. Yields raw LLM output
        strings so callers can forward them as SSE. Cancellation: when the
        caller stops iterating (client disconnect), the httpx response is
        closed and generation is aborted server-side.
        """
        user_prompt = USER_PROMPT_TEMPLATE.format(context=context, query=query, available=available)
        prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"

        if self.groq_enabled and not self._groq_down:
            try:
                print("  -> Streaming Groq generation...")
                async for piece in self._stream_groq(prompt):
                    yield piece
                self.last_was_error = False
                self.generation_engine = "groq"
                return
            except Exception as e:
                print(f"  \u26a0\ufe0f Groq stream unavailable: {e}")
                if self._groq_retryable(e):
                    self._mark_groq_down()

        if self.ollama_enabled:
            try:
                print(f"  -> Streaming local Ollama ({self.ollama_model}) generation...")
                async for piece in self._stream_ollama(prompt):
                    yield piece
                self.last_was_error = False
                self.generation_engine = "ollama"
                return
            except Exception as e:
                print(f"  \u26a0\ufe0f Ollama stream unavailable: {e}")

        # No fallback engine available; nothing to stream.
        self.last_was_error = True
        self.generation_engine = "static"
        return


if __name__ == "__main__":
    router = LLMRouter()

    test_context = "Consumer Protection Act 2019 Section 35: Consumer can file complaint for defective products within 2 years"
    test_query = "I bought a defective phone. What can I do?"

    response = router.generate_response(test_context, test_query)
    print(f"Response: {response}")
