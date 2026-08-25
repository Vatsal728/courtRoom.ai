import os
import sys
import hashlib
import io
import time

import httpx
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "data" / "pdfs"

# Official India Code bitstreams. Query strings (utm params) are intentionally
# stripped. Save names are the canonical filenames used by _map_pdf_act_name.
DOWNLOADS = {
    "BNSS_2023.pdf": "https://www.indiacode.nic.in/bitstream/123456789/21544/1/the_bharatiya_nagarik_suraksha_sanhita,_2023.pdf",
    "BSA_2023.pdf": "https://www.indiacode.nic.in/bitstream/123456789/20063/1/aa202347.pdf",
    "Code_on_Wages_2019.pdf": "https://www.indiacode.nic.in/bitstream/123456789/15793/1/aA2019-29.pdf",
    "Industrial_Relations_Code_2020.pdf": "https://www.indiacode.nic.in/bitstream/123456789/22040/1/A2020-35.pdf",
    "Code_on_Social_Security_2020.pdf": "https://www.indiacode.nic.in/bitstream/123456789/16823/1/aA2020-36.pdf",
    "OSHWC_Code_2020.pdf": "https://www.indiacode.nic.in/bitstream/123456789/22041/1/a2020-37.pdf",
    "Digital_Personal_Data_Protection_Act_2023.pdf": "https://www.indiacode.nic.in/bitstream/123456789/22037/1/a2023-22.pdf",
    "Indian_Contract_Act_1872.pdf": "https://www.indiacode.nic.in/bitstream/123456789/2187/2/A187209.pdf",
}

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) courtRoom.ai law-downloader/1.0"
MIN_SIZE = 10_000  # bytes; any real statute PDF is far larger


def check_reachability(client: httpx.Client, url: str) -> str:
    """HEAD the URL and report HTTP status / content-length when available."""
    try:
        resp = client.head(url, follow_redirects=True, timeout=30)
        length = resp.headers.get("content-length", "?")
        return f"HTTP {resp.status_code} (content-length: {length})"
    except Exception as e:
        return f"HEAD failed: {e}"


def validate_pdf(data: bytes) -> bool:
    return data[:5] == b"%PDF-" and len(data) >= MIN_SIZE


def download_one(client: httpx.Client, name: str, url: str, force: bool = False) -> str:
    dest = PDF_DIR / name
    if dest.exists() and not force:
        return f"SKIP  {name} (already present, {dest.stat().st_size} bytes)"

    reach = check_reachability(client, url)
    print(f"  [{reach}] fetching {name}")

    resp = client.get(url, follow_redirects=True, timeout=180)
    if resp.status_code != 200:
        return f"FAIL  {name} (HTTP {resp.status_code})"
    data = resp.content
    if not validate_pdf(data):
        return f"FAIL  {name} (bad payload: {len(data)} bytes, not a valid PDF)"

    dest.write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()[:12]
    return f"OK    {name} ({len(data)} bytes, sha256:{sha})"


def main():
    force = "--force" in sys.argv
    only = sys.argv[sys.argv.index("--only") + 1:] if "--only" in sys.argv else []

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading to {PDF_DIR}")
    results = []
    with httpx.Client(headers={"User-Agent": USER_AGENT, "Accept": "application/pdf,*/*"}) as client:
        for name, url in DOWNLOADS.items():
            if only and name not in only:
                continue
            results.append(download_one(client, name, url, force))
            time.sleep(0.3)

    print("\n" + "=" * 70)
    print("SUMMARY")
    for r in results:
        print(r)
    failed = [r for r in results if r.startswith("FAIL")]
    print("=" * 70)
    if failed:
        print(f"{len(failed)} download(s) failed. Re-run with --force to retry; "
              "you can also drop the PDFs manually into data/pdfs using the names above.")
        return 1
    print("All downloads OK. Restart the backend (or run build_knowledge_base) to ingest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
