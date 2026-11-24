#!/usr/bin/env python3
"""Versioned repair for E11-V's raw-HTML selector representation.

The original E11-V content artifact is immutable.  This append-only repair
derives canonical visible text from the same archived page bytes, preserving
candidate IDs, ranks, URLs, ordering and cohort membership.
"""
from __future__ import annotations
import hashlib, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/non_jev/v2/e11v_v1_0"
SRC = BASE / "02_selection/V1_COHORT_CONTENT.jsonl"
OUT = BASE / "02_selection/html_text_repair_1"
PARSER_VERSION = "E11V_HTML_VISIBLE_TEXT_BS4_HTML_PARSER_V1"
DROP_TAGS = {"script", "style", "noscript", "template", "svg", "canvas", "nav", "footer", "header", "aside", "form", "input", "button", "select", "textarea"}
BOILERPLATE = re.compile(r"(?:^|[-_\s])(nav|navigation|footer|header|cookie|consent|advert|advertisement|banner|sidebar|breadcrumb|social)(?:$|[-_\s])", re.I)

def sha_bytes(x: bytes) -> str:
    return hashlib.sha256(x).hexdigest()

def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def clean_visible(raw: str) -> tuple[str, str, list[str]]:
    issues: list[str] = []
    if not isinstance(raw, str):
        return "", "invalid_type", ["not_string"]
    if not raw.strip():
        return "", "empty", ["empty_input"]
    looks_html = bool(re.search(r"<\s*(?:!doctype|html|head|body|div|p|article|main|section|script|style)\b", raw, re.I))
    if not looks_html:
        text = re.sub(r"\s+", " ", raw).strip()
        return text, "plain_text", [] if text else ["empty_after_normalization"]
    try:
        from bs4 import BeautifulSoup
    except Exception as exc:
        raise RuntimeError("beautifulsoup4 is required for the pinned parser") from exc
    soup = BeautifulSoup(raw, "html.parser")
    for tag in list(soup.find_all(True)):
        if getattr(tag, "attrs", None) is None:
            continue
        name = (tag.name or "").lower()
        attrs = " ".join([str(tag.get("id") or "")] + [str(x) for x in (tag.get("class") or [])])
        if name in DROP_TAGS or (name in {"div", "section", "span", "p"} and BOILERPLATE.search(attrs)):
            tag.decompose()
    text = soup.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        issues.append("empty_after_html_extraction")
    return text, "html_visible_text", issues

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    source_hash = sha_file(SRC)
    content = OUT / "V1_COHORT_CONTENT_VISIBLE_TEXT.jsonl"
    audit_path = OUT / "V1_HTML_EXTRACTION_AUDIT.jsonl"
    row_count = candidate_count = empty_count = html_count = plain_count = 0
    ids_preserved = ordering_preserved = True
    with SRC.open(encoding="utf-8") as src, content.open("w", encoding="utf-8") as content_out, audit_path.open("w", encoding="utf-8") as audit_out:
      for line in src:
        if not line.strip(): continue
        row = json.loads(line); row_count += 1
        candidates = []
        for d in row["candidates"]:
            raw = str(d.get("content") or "")
            text, status, issues = clean_visible(raw)
            x = dict(d)
            x["content"] = text
            x["raw_content_sha256"] = sha_bytes(raw.encode())
            x["extracted_content_sha256"] = sha_bytes(text.encode())
            x["extraction_status"] = status
            x["extraction_issues"] = issues
            x["extraction_parser"] = PARSER_VERSION
            candidates.append(x)
            candidate_count += 1; empty_count += int(not text); html_count += int(status == "html_visible_text"); plain_count += int(status == "plain_text")
            audit_out.write(json.dumps({"question_id": row["question_id"], "candidate_id": d["candidate_id"], "raw_chars": len(raw), "extracted_chars": len(text), "status": status, "issues": issues, "raw_sha256": x["raw_content_sha256"], "extracted_sha256": x["extracted_content_sha256"]}, ensure_ascii=False, sort_keys=True) + "\n")
        y = dict(row)
        y["candidates"] = candidates
        y["content_representation"] = PARSER_VERSION
        y["source_content_sha256"] = source_hash
        content_out.write(json.dumps(y, ensure_ascii=False, sort_keys=True) + "\n")
        src_ids = [d["candidate_id"] for d in row["candidates"]]
        out_ids = [d["candidate_id"] for d in candidates]
        ids_preserved = ids_preserved and src_ids == out_ids
        ordering_preserved = ordering_preserved and src_ids == out_ids
    summary = {
        "repair_id": "E11V-HTML-TEXT-REPAIR-1",
        "parser_version": PARSER_VERSION,
        "source_path": str(SRC.relative_to(ROOT)),
        "source_sha256": source_hash,
        "output_content_path": str(content.relative_to(ROOT)),
        "output_content_sha256": sha_file(content),
        "audit_path": str(audit_path.relative_to(ROOT)),
        "rows": row_count,
        "candidate_rows": candidate_count,
        "empty_extractions": empty_count,
        "html_rows": html_count,
        "plain_text_rows": plain_count,
        "candidate_ids_preserved": ids_preserved,
        "ordering_preserved": ordering_preserved,
        "gold_or_outcome_fields_used": False,
        "historical_source_modified": False,
    }
    (OUT / "V1_HTML_TEXT_REPAIR_MANIFEST.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0

if __name__ == "__main__": raise SystemExit(main())
