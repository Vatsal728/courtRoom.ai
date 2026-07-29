"""
response_formatter.py - Format legal RAG responses with high accuracy
"""

import re
from typing import List, Dict, Any

class ResponseFormatter:
    """Formatter to standardize responses in the NyayGuru style"""
    
    def __init__(self):
        pass
        
    def format_response(self, query: str, llm_response: str, sources: List[Dict], domain: str, confidence: float) -> Dict[str, Any]:
        """Format raw LLM text and sources into a structured NyayGuru API response"""
        
        # Clean up empty asterisks or broken markdown tokens and translate legacy IPC to BNS 2023
        def clean_markdown(text: str) -> str:
            if not text:
                return text
            # Fix empty bold tokens like '**, Section...' -> 'Section...'
            text = re.sub(r'\*\*\s*,?\s*', '', text)
            # Fix empty numbered lists like '1. **' -> '1. '
            text = re.sub(r'(\d+\.)\s*\*\*\s*', r'\1 ', text)
            # Replace IPC references with BNS 2023 equivalents
            ipc_bns_map = {
                r'\bIPC\s*420\b': 'BNS 2023 Section 318 (formerly IPC 420)',
                r'\bIPC\s*(?:499|500)\b': 'BNS 2023 Section 356 (formerly IPC 499/500)',
                r'\bIPC\s*(?:503|506)\b': 'BNS 2023 Section 351 (formerly IPC 503/506)',
                r'\bIPC\s*(?:378|379)\b': 'BNS 2023 Section 303 (formerly IPC 378/379)',
                r'\bIPC\s*302\b': 'BNS 2023 Section 103 (formerly IPC 302)',
                r'\bCrPC\b': 'BNSS 2023 (formerly CrPC)',
                r'\bIndian Evidence Act\b': 'BSA 2023 (formerly Indian Evidence Act)'
            }
            for pattern, replacement in ipc_bns_map.items():
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            return text.strip()

        # Helper to extract content under markdown headers flexibly
        def extract_section(header_name: str, default_text: str = "") -> str:
            pattern = rf"(?:^|\n)#*\s*\*{{0,2}}{header_name}\*{{0,2}}\s*:?\s*\n?(.*?)(?=\n#+|\n\*\*|$)"
            match = re.search(pattern, llm_response, re.IGNORECASE | re.DOTALL)
            if match:
                res = match.group(1).strip()
                if res:
                    return clean_markdown(res)
            return clean_markdown(default_text)

        # 1. Extract short answer (fallback to full LLM response if empty)
        short_answer = extract_section("SHORT ANSWER", "")
        if not short_answer:
            short_answer = clean_markdown(llm_response)
        
        # 2. Extract legality
        is_this_illegal = extract_section("IS THIS ILLEGAL?", f"Yes, the described actions involve legal provisions under the jurisdiction of {domain.upper()} law and Bharatiya Nyaya Sanhita (BNS) 2023.")
        
        # 3. Extract Criminal Route
        criminal_text = extract_section("CRIMINAL ROUTE")
        criminal_sections = []
        criminal_penalties = []
        criminal_procedure = []
        
        if criminal_text:
            for line in criminal_text.split("\n"):
                line = clean_markdown(line.strip().lstrip("-*• ").strip())
                if not line:
                    continue
                if line.lower().startswith("sections:"):
                    criminal_sections = [clean_markdown(x.strip()) for x in line.split(":", 1)[1].split(",")]
                elif line.lower().startswith("penalties:"):
                    criminal_penalties = [clean_markdown(x.strip()) for x in line.split(":", 1)[1].split(",")]
                elif line.lower().startswith("procedure:"):
                    criminal_procedure = [clean_markdown(x.strip()) for x in line.split(":", 1)[1].split(";")]
                else:
                    if "section" in line.lower() or "bns" in line.lower() or "ipc" in line.lower():
                        criminal_sections.append(line)
                    elif "prison" in line.lower() or "fine" in line.lower() or "punish" in line.lower():
                        criminal_penalties.append(line)
                    else:
                        criminal_procedure.append(line)

        if not criminal_sections:
            criminal_sections = ["BNS 2023 Section 356 (Defamation) / BNS 2023 Section 351 (Criminal Intimidation)"]
        if not criminal_penalties:
            criminal_penalties = ["Imprisonment or fine as specified under BNS 2023"]
        if not criminal_procedure:
            criminal_procedure = [
                "File a First Information Report (FIR) at nearest police station under BNSS 2023",
                "Ensure police registers the crime under appropriate BNS 2023 sections",
                "Cooperatively assist investigation officers with evidence"
            ]

        criminal_route = {
            "applicable_sections": criminal_sections,
            "penalties": criminal_penalties,
            "procedure": criminal_procedure
        }

        # 4. Extract Civil Route
        civil_text = extract_section("CIVIL ROUTE")
        civil_remedies = []
        civil_compensation = "To be claimed based on actual damages"
        civil_procedure = []
        
        if civil_text:
            for line in civil_text.split("\n"):
                line = clean_markdown(line.strip().lstrip("-*• ").strip())
                if not line:
                    continue
                if line.lower().startswith("remedies:"):
                    civil_remedies = [clean_markdown(x.strip()) for x in line.split(":", 1)[1].split(",")]
                elif line.lower().startswith("compensation range:") or line.lower().startswith("compensation:"):
                    civil_compensation = clean_markdown(line.split(":", 1)[1].strip())
                elif line.lower().startswith("procedure:"):
                    civil_procedure = [clean_markdown(x.strip()) for x in line.split(":", 1)[1].split(";")]
                else:
                    if "remedy" in line.lower() or "injunction" in line.lower() or "damages" in line.lower():
                        civil_remedies.append(line)
                    else:
                        civil_procedure.append(line)
                        
        if not civil_remedies:
            civil_remedies = ["Injunction orders", "Declaration of rights", "Compensation claim"]
        if not civil_procedure:
            civil_procedure = [
                "Draft and send a formal legal notice requesting remedy",
                "If unresolved within notice period, file a civil suit in competent court",
                "Seek temporary injunction orders for immediate relief"
            ]

        civil_route = {
            "remedies": civil_remedies,
            "compensation_range": civil_compensation,
            "procedure": civil_procedure
        }

        # 5. Extract Compensation Claims
        compensation_text = extract_section("COMPENSATION CLAIMS")
        compensation_claims = []
        if compensation_text:
            for line in compensation_text.split("\n"):
                line = clean_markdown(line.strip().lstrip("-*• ").strip())
                if line:
                    compensation_claims.append(line)
        if not compensation_claims:
            compensation_claims = [
                "Mental agony and harassment compensation",
                "Direct financial loss reimbursement",
                "Legal costs and suit expenses compensation"
            ]

        # 6. Extract Evidence Needed
        evidence_text = extract_section("EVIDENCE NEEDED")
        evidence_needed = []
        if evidence_text:
            for line in evidence_text.split("\n"):
                line = clean_markdown(line.strip().lstrip("-*• ").strip())
                if line:
                    evidence_needed.append(line)
        if not evidence_needed:
            evidence_needed = [
                "Copy of contract or agreement details",
                "Written notices, emails, or WhatsApp chats between parties",
                "Bank statements showing payments or transactions",
                "Photographic or digital proof of violation if applicable"
            ]

        # 7. Extract Practical Steps (Guaranteed exactly 6 items)
        steps_text = extract_section("PRACTICAL STEPS")
        practical_steps = []
        if steps_text:
            for line in steps_text.split("\n"):
                line = clean_markdown(re.sub(r'^\d+\.\s*', '', line.strip()).lstrip("-*• ").strip())
                if line:
                    practical_steps.append(line)
                    
        default_steps = [
            "Send a formal legal notice outlining claims and giving a 15-day timeline",
            "Collect and organize all physical/digital evidence as per checklist",
            "File an official complaint with the respective local cell (cyber cell, labor board, consumer forum)",
            "Draft a detailed civil plaint with help of a legal professional",
            "Submit the plaint in court and secure an initial hearing date",
            "Follow up on court summons and prepare for trial arguments"
        ]
        
        while len(practical_steps) < 6:
            practical_steps.append(default_steps[len(practical_steps)])
        practical_steps = practical_steps[:6]

        # 8. Extract Applicable Laws mapping
        applicable_laws = {}
        for s in sources:
            act_name = clean_markdown(s.get("act_name") or s.get("source_act") or s.get("source") or "General Law")
            section = clean_markdown(s.get("section_number") or s.get("section") or "General Provision")
            if act_name not in applicable_laws:
                applicable_laws[act_name] = []
            if section not in applicable_laws[act_name]:
                applicable_laws[act_name].append(section)
                
        if not applicable_laws:
            applicable_laws = {"Bharatiya Nyaya Sanhita (BNS) 2023": ["Section 356", "Section 351"]}

        formatted_sources = []
        for s in sources:
            formatted_sources.append({
                "section": clean_markdown(s.get("section_number") or s.get("section") or "Unknown"),
                "topic": clean_markdown(s.get("topic") or "General"),
                "source_act": clean_markdown(s.get("act_name") or s.get("source_act") or s.get("source") or "Unknown Act"),
                "courts": s.get("applicable_courts") or s.get("courts") or ["District Court"],
                "keywords": s.get("keywords") or s.get("relevance_keywords") or [],
                "content_preview": clean_markdown(s.get("content") or s.get("text") or "")
            })

        # Construct the rich NyayGuru Markdown report string
        crim_proc_str = "\n".join([f"  1. {p}" for p in criminal_route['procedure']])
        civ_proc_str = "\n".join([f"  1. {p}" for p in civil_route['procedure']])
        comp_str = "\n".join([f"* {c}" for c in compensation_claims])
        evid_str = "\n".join([f"* {e}" for e in evidence_needed])
        steps_str = "\n".join([f"{i+1}. {step}" for i, step in enumerate(practical_steps)])

        formatted_markdown_report = f"""### 📌 SHORT ANSWER
{short_answer}

### ⚖️ IS THIS ILLEGAL?
{is_this_illegal}

---

### 🚨 CRIMINAL ROUTE
* **Applicable Sections:** {', '.join(criminal_route['applicable_sections'])}
* **Penalties:** {', '.join(criminal_route['penalties'])}
* **Procedure:**
{crim_proc_str}

---

### 📜 CIVIL ROUTE
* **Remedies:** {', '.join(civil_route['remedies'])}
* **Compensation Range:** {civil_route['compensation_range']}
* **Procedure:**
{civ_proc_str}

---

### 💰 COMPENSATION CLAIMS
{comp_str}

---

### 📁 EVIDENCE CHECKLIST
{evid_str}

---

### 📋 PRACTICAL STEPS (ACTION PLAN)
{steps_str}"""

        formatted_markdown_report = clean_markdown(formatted_markdown_report)

        return {
            "query": query,
            "response_type": domain,
            "confidence_score": confidence,
            "short_answer": short_answer,
            "full_response": formatted_markdown_report,
            "response": formatted_markdown_report,
            "is_this_illegal": is_this_illegal,
            "criminal_route": criminal_route,
            "civil_route": civil_route,
            "practical_steps": practical_steps,
            "compensation_claims": compensation_claims,
            "evidence_needed": evidence_needed,
            "applicable_laws": applicable_laws,
            "sources": formatted_sources,
            "status": "success"
        }

def format_legal_response(query: str, llm_response: str, sources: List[Dict], domain: str, confidence: float) -> Dict[str, Any]:
    """Compatibility function mapping format_legal_response requests"""
    formatter = ResponseFormatter()
    return formatter.format_response(query, llm_response, sources, domain, confidence)
