import httpx, json, time

url = "http://localhost:8000/query/stream"
payload = {"query": "ભારતમાં ક્રેડિટ કાર્ડ ધારણ કરવા માટે કયા કાયદા લાગુ પડે છે?", "language": "gu"}

start = time.monotonic()
with httpx.stream("POST", url, json=payload, timeout=120) as resp:
    for line in resp.iter_lines():
        if line.strip():
            if "final" in line:
                data = json.loads(line.replace("data: ", ""))
                elapsed = time.monotonic() - start
                print(f"Total: {elapsed:.1f}s")
                result = data
                print("translated:", result.get("translated"))
                print("translation_corrected:", result.get("translation_corrected"))
                print("query_language:", result.get("query_language"))
                print("short_answer len:", len(result.get("short_answer", "")))
                print("is_this_illegal len:", len(result.get("is_this_illegal", "")))
                print("full_response len:", len(result.get("full_response", "")))
                print("has original_short_answer:", "original_short_answer" in result)
                sources = result.get("sources", [])
                if sources:
                    print("sources[0] section_title len:", len(sources[0].get("section_title", "")))
                    st = sources[0].get("section_title", "")
                    has_gu = any(ord(c) > 127 for c in st)
                    print("sources[0] has gu text:", has_gu)
                break
            elif "status" in line and "translating" in line:
                print(f"[{time.monotonic()-start:.1f}s] translation started")
