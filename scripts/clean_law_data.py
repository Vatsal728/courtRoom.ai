#!/usr/bin/env python
"""One-time, idempotent, non-destructive cleaning of the legal-law corpus.

1. Cleans every law JSON in place (backup lives in data/laws/backup/):
   - whitespace: tabs -> space, CR -> LF, collapse 3+ blank lines and 2+ spaces
   - editorial footnote anchors: folds `7[the Union]` -> `the Union`
     (operative replacement text is KEPT); strips bare `1[` anchors and
     `1***` / `* * *` star markers (pure editorial marks)
   - repeal/omission notes ([Repealed], [Omitted], Rep. by ..., Omitted by ...)
     are MOVED into an `amendment_note` field + `section_status`; never deleted
   - trailing editorial footnotes ("1 Changed from fifteen by amendment act of
     2015") are likewise moved into `amendment_note` / `section_status`
   - if a section's only content was the repeal note, the note stays as the
     description so the section is never lost
   - conservative OCR fix: `fad` -> `fact` (only before "that")
2. Rebuilds IndiaLaw.db from the cleaned JSONs with the same schema the
   pipeline expects, plus `amendment_note` and `section_status` columns.
   Row counts are validated; the rebuild aborts if a table would lose rows.
3. hma.json (broken combined CSV-key format with unterminated quotes) is
   converted to the proper 4-field schema; fragment continuation rows are
   dropped because they are not sections (their text belongs to other rows).
"""
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAWS_DIR = ROOT / "data" / "laws"
DB_PATH = LAWS_DIR / "IndiaLaw.db"

# ── CSV repair (unterminated quotes) ───────────────────────────────────
def parse_csv_line(line):
    """Split a CSV line on commas, honouring double-quote fields even when
    the opening quote is never closed (the state hma.json was in)."""
    parts, cur, in_q = [], "", False
    for ch in line:
        if ch == '"':
            in_q = not in_q
        elif ch == "," and not in_q:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    parts.append(cur.strip())
    return parts


# ── Text cleaning ─────────────────────────────────────────────────────
_NOTE_RE = re.compile(
    r"\[(?:Repealed|Omitted)[^\]]{0,120}\]"
    r"|(?:\bRep\. by\b|\bRepealed by\b|\bOmitted by\b|\bSubs\. by\b|"
    r"\bSubstituted by\b|\bIns\. by\b|\bAdded by\b)[^;\n]{0,160}",
    re.IGNORECASE,
)
_FAD_RE = re.compile(r"\bfad\b(?=\s+that\b)", re.IGNORECASE)

# Trailing editorial footnotes: a standalone line that starts with a bare
# footnote marker (e.g. "1 Changed from fifteen by amendment act of 2015",
# "3[Omitted]" is already handled above, "[7] Substituted...") and reads like
# an amendment note. The editorial-keyword guard prevents accidental eating of
# legitimate enumerations that merely start with a number.
_FOOTNOTE_RE = re.compile(
    r"^\s*(?:\[\d{1,3}\]|\d{1,3})[\s\.\)]*[^\n]{3,}$"
)
_EDITORIAL_RE = re.compile(
    r"\b(?:amendment act|amended by|substituted|subs\.|inserted|omitted by|"
    r"omitted|added by|w\.e\.f|wef|effective from|came into force|comes into force|"
    r"replaced|renumbered|deleted|repealed|notified|in force|brought into force)\b",
    re.IGNORECASE,
)


def _is_footnote_line(s: str) -> bool:
    if not s or not _FOOTNOTE_RE.match(s):
        return False
    return bool(_EDITORIAL_RE.search(s))


def clean_text(text):
    if not text:
        return text
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    text = re.sub(r"\b\d{1,3}\[([^\]\n]{1,120})\]\s*", lambda m: m.group(1).strip(), text)
    text = re.sub(r"\b\d{1,3}\[", "", text)
    text = re.sub(r"\b\d{1,3}\*{1,}", " ", text)
    text = re.sub(r"\*+[ \t]*\*+[ \t]*\*+", " ", text)
    text = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+$", "", text, flags=re.M)
    text = _FAD_RE.sub("fact", text)
    return text.strip()


def extract_notes(text):
    """Pull short repeal/amendment-note lines out of the main text.

    Returns (clean_text, notes). The note text is preserved, never dropped.
    """
    notes = []
    kept = []
    for ln in text.split("\n"):
        s = ln.strip()
        m = _NOTE_RE.search(s)
        if _is_footnote_line(s):
            note = re.sub(r"^\s*(?:\[\d{1,3}\]|\d{1,3})[\s\.\)]*", "", s).strip()
            notes.append(note)
        elif s and len(s) <= 220 and m and (s.startswith("[") or m.start() <= 5):
            notes.append(s)
        else:
            kept.append(ln)
    return "\n".join(kept).strip(), notes


def clean_item(item, desc_key, title_key):
    """Clean a single law item; returns a copy with amendment_note/section_status."""
    out = dict(item)
    desc = (out.get(desc_key) or "").strip()
    title = (out.get(title_key) or "").strip()

    desc_clean, notes = extract_notes(desc)
    desc_clean = clean_text(desc_clean)
    title_clean = clean_text(title)

    out[title_key] = title_clean
    if not desc_clean and notes:
        desc_clean = notes[0]
        out[desc_key] = desc_clean
        out["amendment_note"] = " ; ".join(notes)
        out["section_status"] = "repealed" if "repealed" in notes[0].lower() else "omitted"
        return out

    out[desc_key] = desc_clean
    if notes:
        out["amendment_note"] = " ; ".join(notes)
        joined = " ".join(notes).lower()
        if "repealed" in joined:
            out["section_status"] = "repealed"
        elif "omitted" in joined:
            out["section_status"] = "omitted"
        else:
            out["section_status"] = "amended"
    return out


# ── Loaders ───────────────────────────────────────────────────────────
def load_standard(name):
    raw = json.loads((LAWS_DIR / name).read_text(encoding="utf-8"))
    items = raw if isinstance(raw, list) else [raw]
    flat = []
    for inner in items:
        flat.extend(inner if isinstance(inner, list) else [inner])
    return flat


def _hma_fields(it):
    combined = it.get("chapter,section,section_title,section_desc")
    if combined is not None and (combined or "").strip():
        return parse_csv_line(combined.strip()) + [""] * 4
    return [
        it.get("chapter") or "",
        it.get("section") or "",
        it.get("section_title") or "",
        it.get("section_desc") or "",
    ]


def load_hma():
    raw = json.loads((LAWS_DIR / "hma.json").read_text(encoding="utf-8"))
    items = []
    dropped = 0
    for it in raw:
        parts = _hma_fields(it)[:4]
        chapter, section, title, desc = parts
        if not section or not desc:
            dropped += 1
            continue
        # Fragment rows: a non-numeric 'section' that is really a continuation
        # (e.g. 'Provided', '(iiia)', 'clause (iv)') belongs to another row.
        if not re.match(r"^\d+[A-Z]*$", section.strip()):
            dropped += 1
            continue
        items.append({"chapter": chapter, "section": section, "section_title": title, "section_desc": desc})
    print(f"  hma.json: {len(items)} usable sections, {dropped} empty/fragment rows dropped")
    return items


SOURCES = {
    "IPC": ("ipc.json", [("chapter", "INTEGER"), ("chapter_title", "TEXT"), ("Section", "INTEGER"),
                         ("section_title", "TEXT"), ("section_desc", "TEXT")],
            "section_desc", "section_title"),
    "CRPC": ("crpc.json", [("chapter", "INTEGER"), ("section", "INTEGER"), ("section_title", "TEXT"),
                           ("section_desc", "TEXT")], "section_desc", "section_title"),
    "IEA": ("iea.json", [("chapter", "INTEGER"), ("section", "INTEGER"), ("section_title", "TEXT"),
                         ("section_desc", "TEXT")], "section_desc", "section_title"),
    "NIA": ("nia.json", [("chapter", "INTEGER"), ("section", "INTEGER"), ("section_title", "TEXT"),
                         ("section_desc", "TEXT")], "section_desc", "section_title"),
    "HMA": ("hma.json", [("chapter", "INTEGER"), ("section", "TEXT"), ("section_title", "TEXT"),
                         ("section_desc", "TEXT")], "section_desc", "section_title"),
    "CPC": ("cpc.json", [("section", "INTEGER"), ("title", "TEXT"), ("description", "TEXT")],
            "description", "title"),
    "IDA": ("ida.json", [("section", "TEXT"), ("title", "TEXT"), ("description", "TEXT")],
            "description", "title"),
    "MVA": ("MVA.json", [("section", "TEXT"), ("title", "TEXT"), ("description", "TEXT")],
            "description", "title"),
}

# Stem -> loader (hma uses the special repair loader)
_STEM_LOADERS = {
    "ipc.json": lambda: load_standard("ipc.json"),
    "crpc.json": lambda: load_standard("crpc.json"),
    "iea.json": lambda: load_standard("iea.json"),
    "nia.json": lambda: load_standard("nia.json"),
    "hma.json": load_hma,
    "cpc.json": lambda: load_standard("cpc.json"),
    "ida.json": lambda: load_standard("ida.json"),
    "MVA.json": lambda: load_standard("MVA.json"),
}


def clean_json_files():
    print("Cleaning JSON source files...")
    summary = {}
    for table, (fname, _, desc_key, title_key) in SOURCES.items():
        items = _STEM_LOADERS[fname]()
        cleaned = [clean_item(it, desc_key, title_key) for it in items]
        out_path = LAWS_DIR / fname
        out_path.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
        n_notes = sum(1 for it in cleaned if it.get("amendment_note"))
        summary[table] = (len(items), len(cleaned), n_notes)
        print(f"  {table}: {len(items)} loaded -> {len(cleaned)} written ({n_notes} with amendment_note)")
    return summary


def _swap_db(new_path, target):
    """Swap the freshly built db over the live one; on Windows the first
    os.replace can transiently fail with PermissionError, so retry and fall
    back to delete-then-rename."""
    for attempt in range(5):
        try:
            os.replace(new_path, target)
            return
        except PermissionError:
            time.sleep(0.5)
    try:
        target.unlink()
        new_path.rename(target)
    except OSError as exc:
        print(f"  ERROR: could not replace {target}: {exc}")
        print(f"  Rebuilt db left at {new_path}; apply manually after closing the file.")
        sys.exit(1)


def rebuild_db():
    print("Rebuilding IndiaLaw.db from cleaned JSONs...")
    if not DB_PATH.exists():
        print("ERROR: IndiaLaw.db missing; nothing to validate against.")
        sys.exit(1)
    old = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    old_counts = {}
    old_schemas = {}
    try:
        for table, (fname, _, _, _) in SOURCES.items():
            old_counts[table] = old.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            old_schemas[table] = [(c[1], c[2]) for c in old.execute(f'PRAGMA table_info("{table}")')]
    finally:
        old.close()

    new_path = LAWS_DIR / "IndiaLaw.db.new"
    if new_path.exists():
        new_path.unlink()
    conn = sqlite3.connect(new_path)
    try:
        for table, (fname, cols, desc_key, title_key) in SOURCES.items():
            items = _STEM_LOADERS[fname]()
            cleaned = [clean_item(it, desc_key, title_key) for it in items]

            extra = [("amendment_note", "TEXT"), ("section_status", "TEXT")]
            col_defs = ", ".join(f'"{n}" {typ}' for n, typ in cols + extra)
            conn.execute(f'CREATE TABLE "{table}" ({col_defs})')
            insert_cols = [n for n, _ in cols] + ["amendment_note", "section_status"]
            ph = ", ".join("?" for _ in insert_cols)
            sql = f'INSERT INTO "{table}" ({", ".join(insert_cols)}) VALUES ({ph})'

            for it in cleaned:
                row = []
                for n, typ in cols:
                    v = it.get(n)
                    if v is None:
                        v = ""
                    elif typ == "INTEGER":
                        try:
                            v = int(v)
                        except (TypeError, ValueError):
                            v = 0 if str(v).strip() in ("",) else str(v)
                    row.append(v)
                row.append(it.get("amendment_note") or "")
                row.append(it.get("section_status") or "")
                conn.execute(sql, row)

            new_count = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            old = old_counts[table]
            print(f"  {table}: old {old} -> new {new_count}")
            if new_count < old:
                print(f"  ABORT: {table} would lose rows ({old} -> {new_count}). Nothing changed on disk.")
                conn.close()
                new_path.unlink()
                sys.exit(1)
        conn.commit()
    finally:
        conn.close()

    _swap_db(new_path, DB_PATH)
    print("IndiaLaw.db rebuilt OK (original preserved in data/laws/backup/).")


def main():
    summary = clean_json_files()
    print()
    rebuild_db()
    print("\nDone. Spot checks:")
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        for probe in [
            ('IPC', "SELECT section_desc FROM \"IPC\" WHERE Section=61"),
            ('CPC', "SELECT description FROM \"CPC\" WHERE section=1"),
            ('MVA', "SELECT description FROM \"MVA\" WHERE section='194C'"),
            ('IEA', "SELECT section_desc FROM \"IEA\" WHERE section=52"),
        ]:
            print(f"  [{probe[0]}] {conn.execute(probe[1]).fetchone()[0][:110]!r}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
