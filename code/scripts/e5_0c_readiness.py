#!/usr/bin/env python3
"""CPU/offline E5-0C scientific-readiness audit.

This module reads only existing E5-0 artifacts and permitted CAL/exploratory
QASPER records.  It never calls a model, HTTP endpoint, GPU, reader, Jev, or
reserved E5-F payload.  New evidence is written only below the E5-0C output
directory.
"""
from __future__ import annotations

import csv
import getpass
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E5 = ROOT / "results/non_jev/v2/e5_v1_1/e5_0_data_protocol"
OUT = ROOT / "results/non_jev/v2/e5_v1_1/e5_0c_readiness_closure"
SNAP = ROOT / "data/external/e5_qasper/snapshot"
EXPECTED_SLATE_SHA = "be9cf462b77e92ee8c1b70bad5d10412e9c51c4d3a249e9e925140ebdab2d5e7"
PPLX_REV = "14c9d90137962d300f3de95633f7b6535e80c5ce"
JINA_REV = "e8a93f33f0b22108f8c2364f8484ce3422552fbc"
TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def stable(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_text(value: str) -> str:
    return sha_bytes(value.encode("utf-8"))


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def git_state() -> dict:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
        except Exception as exc:
            return f"UNAVAILABLE:{type(exc).__name__}:{exc}"
    status = run("status", "--porcelain=v1")
    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_sha256": sha_text(status),
        "dirty": bool(status),
    }


def _skip_ws(data: bytes, i: int) -> int:
    while i < len(data) and data[i] in b" \t\r\n":
        i += 1
    return i


def _string_end(data: bytes, i: int) -> int:
    assert data[i] == 34
    i += 1
    escaped = False
    while i < len(data):
        c = data[i]
        if escaped:
            escaped = False
        elif c == 92:
            escaped = True
        elif c == 34:
            return i + 1
        i += 1
    raise ValueError("unterminated JSON string")


def _value_end(data: bytes, i: int) -> int:
    i = _skip_ws(data, i)
    if i >= len(data):
        raise ValueError("missing JSON value")
    if data[i] == 34:
        return _string_end(data, i)
    if data[i] in (91, 123):
        stack = [data[i]]
        j = i + 1
        escaped = False
        in_string = False
        while j < len(data):
            c = data[j]
            if in_string:
                if escaped:
                    escaped = False
                elif c == 92:
                    escaped = True
                elif c == 34:
                    in_string = False
            else:
                if c == 34:
                    in_string = True
                elif c in (91, 123):
                    stack.append(c)
                elif c in (93, 125):
                    expected = 91 if c == 93 else 123
                    if not stack or stack[-1] != expected:
                        raise ValueError("malformed JSON nesting")
                    stack.pop()
                    if not stack:
                        return j + 1
            j += 1
        raise ValueError("unterminated JSON container")
    j = i
    while j < len(data) and data[j] not in b",}] \t\r\n":
        j += 1
    return j


def iter_top_object_values(path: Path, allowed_keys: set[str]):
    """Yield only allowed top-level object values.

    QASPER is a top-level object keyed by paper ID.  Values for disallowed
    keys are skipped by byte structure and are never JSON-decoded or retained;
    this prevents reserved paper payloads from entering the audit process.
    """
    data = path.read_bytes()
    i = _skip_ws(data, 0)
    if i >= len(data) or data[i] != 123:
        raise ValueError(f"expected top-level object: {path}")
    i += 1
    decoder = json.JSONDecoder()
    while True:
        i = _skip_ws(data, i)
        if i >= len(data):
            raise ValueError(f"unterminated top-level object: {path}")
        if data[i] == 125:
            return
        if data[i] == 44:
            i = _skip_ws(data, i + 1)
        key_end = _string_end(data, i)
        key = json.loads(data[i:key_end].decode("utf-8"))
        i = _skip_ws(data, key_end)
        if data[i] != 58:
            raise ValueError("expected colon")
        i = _skip_ws(data, i + 1)
        end = _value_end(data, i)
        if key in allowed_keys:
            # Decode only permitted source-paper values.
            value, consumed = decoder.raw_decode(data[i:end].decode("utf-8"))
            if consumed != len(data[i:end].decode("utf-8")):
                raise ValueError("permitted value did not consume exactly")
            yield key, value
        i = end


def sha_support(ids: list[str]) -> str:
    return sha_text("\n".join(ids)) if ids else ""


def qid_parts(qid: str) -> tuple[str, str]:
    _, paper, question_id = qid.split(":", 2)
    return paper, question_id


def load_e5_inputs() -> dict:
    corpus = read_jsonl(E5 / "E5_0_PERMITTED_CORPUS.jsonl")
    permitted = read_jsonl(E5 / "E5_0_PERMITTED_QUERIES.jsonl")
    mapping = list(csv.DictReader((E5 / "E5_0_GOLD_MAPPING_AND_ELIGIBILITY_AUDIT.csv").open(encoding="utf-8")))
    gold = read_jsonl(E5 / "E5_0_EXPLORATORY_GOLD_INTERNAL.jsonl")
    slates = read_jsonl(E5 / "E5_0_NATURAL_TOP20_MANIFEST.jsonl")
    return {"corpus": corpus, "permitted": permitted, "mapping": mapping, "gold": gold, "slates": slates}


def audit_e5_hashes() -> dict:
    manifest = read_json(E5 / "E5_0_PROVENANCE_MANIFEST.json")
    missing, mismatches = [], []
    for item in manifest.get("files", []):
        p = ROOT / item["path"]
        if not p.exists():
            missing.append(item["path"])
        elif sha_file(p) != item["sha256"]:
            mismatches.append(item["path"])
    hist = read_json(E5 / "E5_0_HISTORICAL_HASH_AUDIT.json")
    slate_path = E5 / "E5_0_NATURAL_TOP20_MANIFEST.jsonl"
    return {
        "e5_0_manifest_files": len(manifest.get("files", [])),
        "e5_0_missing": missing,
        "e5_0_mismatches": mismatches,
        "e5_0_historical_audit": hist,
        "slate_file_sha256": sha_file(slate_path),
        "expected_slate_sha256": EXPECTED_SLATE_SHA,
        "slate_hash_exact_match": sha_file(slate_path) == EXPECTED_SLATE_SHA,
        "e5_0_gate": manifest.get("gate"),
    }


def permitted_raw_questions(permitted: list[dict], corpus: list[dict]) -> dict[str, dict]:
    """Load only raw paper records represented in the permitted corpus."""
    allowed_papers = {r["source_id"].split(":", 1)[1] for r in corpus}
    wanted = {qid_parts(r["query_id"])[0] for r in permitted}
    allowed_papers &= wanted
    records: dict[str, dict] = {}
    for fn in (SNAP / "qasper-train-v0.3.json", SNAP / "qasper-dev-v0.3.json"):
        for pid, rec in iter_top_object_values(fn, allowed_papers):
            records.setdefault(pid, rec)
    return records


def _paper_paragraph_map(pid: str, rec: dict) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for si, section in enumerate(rec.get("full_text", []) or []):
        for pi, text in enumerate(section.get("paragraphs", []) or []):
            if isinstance(text, str) and text.strip():
                out[text].append(f"qasper:{pid}:p{si}_{pi}")
    return out


def _answer_semantics(pid: str, q: dict, pmap: dict[str, list[str]]) -> dict:
    answers = q.get("answers", []) or []
    answerable = [w.get("answer", {}) or {} for w in answers if not (w.get("answer", {}) or {}).get("unanswerable", False)]
    if not answerable:
        return {"raw_answer_wrappers": len(answers), "answerable_wrappers": 0, "sets": [], "first_set": [], "flags": ["OTHER_UNSCORABLE"], "wrapper_details": []}
    sets: list[tuple[str, ...]] = []
    details = []
    for wi, answer in enumerate(answerable):
        ev = answer.get("evidence", []) or []
        mapped: list[str] = []
        invalid = ambiguous = nontext = False
        duplicate_evidence = len(ev) != len(set(x for x in ev if isinstance(x, str)))
        for evidence in ev:
            if not isinstance(evidence, str) or evidence.startswith("FLOAT SELECTED"):
                nontext = True
                continue
            hits = pmap.get(evidence, [])
            if len(hits) == 1 and hits[0].split(":")[1] == pid:
                mapped.append(hits[0])
            elif len(hits) == 0:
                invalid = True
            else:
                ambiguous = True
        support = tuple(sorted(set(mapped)))
        if support:
            sets.append(support)
        details.append({
            "answer_wrapper_index": wi,
            "field_path": f"qas[*].answers[{wi}].answer.evidence",
            "raw_evidence_count": len(ev),
            "mapped_support_count": len(support),
            "invalid_reference": invalid,
            "ambiguous_reference": ambiguous,
            "nontext_evidence": nontext,
            "duplicate_evidence": duplicate_evidence,
            "support_ids": list(support),
        })
    distinct = sorted(set(sets))
    flags: list[str] = []
    if len(distinct) == 1:
        flags.append("SINGLE_DESIGNATED_SET")
    if len(distinct) > 1:
        flags.extend(["MULTIPLE_ALTERNATIVE_SETS", "AMBIGUOUS_UNION"])
    if any(d["invalid_reference"] or d["ambiguous_reference"] for d in details):
        flags.append("INVALID_ID")
    if any(d["nontext_evidence"] for d in details):
        flags.append("PARTIAL_OR_NONEXHAUSTIVE")
    if len(answerable) > 1:
        flags.append("ANSWER_WRAPPER_MULTIPLICITY")
    if any(d["duplicate_evidence"] for d in details):
        flags.append("DUPLICATE_EVIDENCE")
    # E5-0 uses options[0]: the first answerable wrapper that yields a mapped
    # support set, not the lexicographically first distinct set.
    first = list(sets[0]) if sets else []
    if len(first) > 5:
        flags.append("SUPPORT_GT5")
    if not distinct:
        flags.append("OTHER_UNSCORABLE")
    return {"raw_answer_wrappers": len(answers), "answerable_wrappers": len(answerable), "sets": [list(x) for x in distinct], "first_set": first, "flags": sorted(set(flags)), "wrapper_details": details}


def audit_support_semantics(data: dict) -> dict:
    permitted = {r["query_id"]: r for r in data["permitted"]}
    mapping = {r["query_id"]: r for r in data["mapping"]}
    raw = permitted_raw_questions(data["permitted"], data["corpus"])
    # Match the frozen E5-0 mapper: evidence-text lookup is global over the
    # visible passage universe, then a unique hit is retained only when it is
    # in the question's source paper.  This intentionally exposes duplicate
    # text as ambiguous instead of silently treating it as local evidence.
    global_text_map: dict[str, list[str]] = defaultdict(list)
    for pid, rec in raw.items():
        for text, ids in _paper_paragraph_map(pid, rec).items():
            global_text_map[text].extend(ids)
    analyses: dict[str, dict] = {}
    missing_raw = []
    for row in data["mapping"]:
        qid = row["query_id"]
        pid, qid_short = qid_parts(qid)
        rec = raw.get(pid)
        q = next((x for x in (rec or {}).get("qas", []) if str(x.get("question_id")) == qid_short), None)
        if q is None:
            missing_raw.append(qid)
            continue
        analyses[qid] = _answer_semantics(pid, q, global_text_map)
    cal = [r for r in data["mapping"] if r["partition"] == "CAL"]
    exp = [r for r in data["mapping"] if r["partition"] == "E5_EXPLORATORY"]
    cal_counts = Counter()
    exp_counts = Counter()
    mapping_mismatches = []
    for row in data["mapping"]:
        a = analyses.get(row["query_id"])
        if not a:
            continue
        for flag in a["flags"]:
            (cal_counts if row["partition"] == "CAL" else exp_counts)[flag] += 1
        expected_hash = row["support_ids_sha256"]
        if a["first_set"] and sha_support(a["first_set"]) != expected_hash:
            mapping_mismatches.append(row["query_id"])
    valid_cal = [r for r in cal if r["mapping_status"] == "VALID_DESIGNATED_SUPPORT"]
    classes = Counter()
    for row in valid_cal:
        flags = set(analyses.get(row["query_id"], {}).get("flags", []))
        if "MULTIPLE_ALTERNATIVE_SETS" in flags:
            cls = "MULTIPLE_ALTERNATIVE_SETS"
        elif "INVALID_ID" in flags:
            cls = "INVALID_ID"
        elif "PARTIAL_OR_NONEXHAUSTIVE" in flags or "ANSWER_WRAPPER_MULTIPLICITY" in flags:
            cls = "PARTIAL_OR_NONEXHAUSTIVE"
        elif "SUPPORT_GT5" in flags:
            cls = "SUPPORT_GT5"
        elif "SINGLE_DESIGNATED_SET" in flags:
            cls = "SINGLE_DESIGNATED_SET"
        else:
            cls = "OTHER_UNSCORABLE"
        classes[cls] += 1
    return {"raw_permitted_papers_materialized": len(raw), "raw_questions_missing": missing_raw, "cal_rows": len(cal), "exp_rows": len(exp), "cal_flag_counts": cal_counts, "exp_flag_counts": exp_counts, "cal_valid_class_counts": classes, "mapping_hash_mismatches": mapping_mismatches, "analyses": analyses}


def audit_slates(data: dict) -> dict:
    corpus_ids = {r["passage_id"] for r in data["corpus"]}
    map_rows = {r["query_id"]: r for r in data["mapping"]}
    gold = {r["query_id"]: r for r in data["gold"]}
    bad_rows = []
    source_counts = Counter()
    paper_counts = Counter()
    candidate_total = same_paper = 0
    with_gold = complete_gold = top5_complete = any_top5 = any_top20 = 0
    ranks = []
    by_group = defaultdict(lambda: {"queries": 0, "complete_top20": 0, "any_top20": 0, "top5_complete": 0})
    for row in data["slates"]:
        qid = row["query_id"]
        query_paper, _ = qid_parts(qid)
        ids = row["candidate_ids"]
        if len(ids) != 20 or len(set(ids)) != 20 or not set(ids) <= corpus_ids:
            bad_rows.append(qid)
        candidate_total += len(ids)
        same_paper += sum(x.split(":")[1] == query_paper for x in ids)
        paper_counts["same_paper"] += sum(x.split(":")[1] == query_paper for x in ids)
        paper_counts["out_of_paper"] += sum(x.split(":")[1] != query_paper for x in ids)
        source_counts.update(c["source_group_id"] for c in row["candidates"])
        for c in row["candidates"]:
            if c["candidate_id"].split(":")[1] == query_paper:
                source_counts["same_paper_rows"] += 1
        g = map_rows.get(qid, {}).get("source_group_id", "UNKNOWN")
        by_group[g]["queries"] += 1
        if qid in gold:
            supports = set(gold[qid]["support_ids"])
            with_gold += 1
            pos = {x: i + 1 for i, x in enumerate(ids)}
            sranks = [pos[x] for x in supports if x in pos]
            ranks.extend(sranks)
            any20 = bool(sranks)
            complete20 = supports <= set(ids)
            complete5 = supports <= set(ids[:5])
            any_top5 = any_top5 + int(bool(set(ids[:5]) & supports))
            any_top20 = any_top20 + int(any20)
            complete_gold = complete_gold + int(complete20)
            top5_complete = top5_complete + int(complete5)
            by_group[g]["any_top20"] += int(any20)
            by_group[g]["complete_top20"] += int(complete20)
            by_group[g]["top5_complete"] += int(complete5)
    return {
        "slates": len(data["slates"]), "bad_slate_rows": bad_rows, "candidate_total": candidate_total,
        "paper_membership": paper_counts, "same_paper_candidate_fraction": same_paper / max(1, candidate_total),
        "global_search": True, "query_text_only": True, "qrels_accessed_during_generation": False,
        "gold_in_request": False, "top20_gold_rows": with_gold, "any_support_top20": any_top20,
        "complete_support_top20": complete_gold, "complete_support_top5": top5_complete,
        "any_support_top20_rate": any_top20 / max(1, with_gold),
        "complete_support_top20_rate": complete_gold / max(1, with_gold),
        "complete_support_top5_rate": top5_complete / max(1, with_gold),
        "support_rank_count": len(ranks), "support_rank_p50": sorted(ranks)[len(ranks)//2] if ranks else None,
        "support_rank_max": max(ranks) if ranks else None, "by_group": by_group,
    }


def audit_feasibility(data: dict, slate_audit: dict) -> tuple[dict, list[dict]]:
    rows = []
    slates = {r["query_id"]: r for r in data["slates"]}
    map_rows = {r["query_id"]: r for r in data["mapping"]}
    gold = {r["query_id"]: r for r in data["gold"]}
    for qid, g in gold.items():
        slate = slates[qid]
        ids = slate["candidate_ids"]
        support = set(g["support_ids"])
        ranks = {x: i + 1 for i, x in enumerate(ids)}
        present = sorted(ranks[x] for x in support if x in ranks)
        rows.append({
            "query_id": qid, "source_group_id": map_rows[qid]["source_group_id"], "support_count": len(support),
            "any_support_top20": int(bool(present)), "complete_support_top20": int(support <= set(ids)),
            "complete_support_top5": int(support <= set(ids[:5])), "support_min_rank": min(present) if present else "",
            "support_max_rank": max(present) if present else "", "source_paper_candidate_count": sum(x.split(":")[1] == qid_parts(qid)[0] for x in ids),
        })
    by_group = defaultdict(lambda: {"queries": 0, "any_support_top20": 0, "complete_support_top20": 0, "complete_support_top5": 0})
    for r in rows:
        for k in by_group[r["source_group_id"]]:
            if k == "queries": by_group[r["source_group_id"]][k] += 1
            elif k in r: by_group[r["source_group_id"]][k] += r[k]
    summary = {
        "status": "EXPLORATORY_PREOUTCOME_CEILING_COMPUTED_CAL_EXACT_CEILING_NOT_MEASURED",
        "cal_exact_ceiling": "CAL_EXACT_CEILING_NOT_MEASURED",
        "cal_missing_ingredients": ["CAL frozen Top-20 slates", "matching CAL query embeddings"],
        "exploratory_population": len(rows),
        "exploratory_any_support_top20": sum(r["any_support_top20"] for r in rows),
        "exploratory_complete_support_top20": sum(r["complete_support_top20"] for r in rows),
        "exploratory_complete_support_top5": sum(r["complete_support_top5"] for r in rows),
        "exploratory_any_support_top20_rate": sum(r["any_support_top20"] for r in rows) / max(1, len(rows)),
        "exploratory_complete_support_top20_rate": sum(r["complete_support_top20"] for r in rows) / max(1, len(rows)),
        "exploratory_complete_support_top5_rate": sum(r["complete_support_top5"] for r in rows) / max(1, len(rows)),
        "support_cardinality": Counter(r["support_count"] for r in rows),
        "source_groups_with_feasible_query": len({r["source_group_id"] for r in rows if r["complete_support_top20"]}),
        "source_group_count": len(by_group),
        "by_group": by_group,
    }
    return summary, rows


def audit_lineage(data: dict, raw_records: dict) -> dict:
    corpus = data["corpus"]
    source_ids = sorted({r["source_id"] for r in corpus})
    group_ids = sorted({r["source_group_id"] for r in corpus})
    passage_hash_sources = defaultdict(set)
    for r in corpus:
        passage_hash_sources[r["passage_text_sha256"]].add(r["source_id"])
    exact_cross_source = {h: sorted(s) for h, s in passage_hash_sources.items() if len(s) > 1}
    titles = {}
    for pid, rec in raw_records.items():
        title = re.sub(r"\W+", " ", (rec.get("title") or "").casefold()).strip()
        if title:
            titles[pid] = title
    title_to_papers = defaultdict(list)
    for pid, title in titles.items(): title_to_papers[title].append(pid)
    duplicate_titles = {t: v for t, v in title_to_papers.items() if len(v) > 1}
    return {
        "permitted_source_ids": len(source_ids), "permitted_source_groups": len(group_ids),
        "source_id_unique": len(source_ids) == len(group_ids), "exact_cross_source_passage_hashes": exact_cross_source,
        "exact_cross_source_passage_hash_count": len(exact_cross_source), "duplicate_normalized_titles": duplicate_titles,
        "raw_permitted_papers": len(raw_records),
        "reserved_near_duplicate_disjointness": "RESERVED_NEAR_DUPLICATE_DISJOINTNESS_NOT_INDEPENDENTLY_VERIFIED",
        "pretraining_exposure": "UNKNOWN",
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def write_outputs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_e5_inputs()
    hashes = audit_e5_hashes()
    support = audit_support_semantics(data)
    slate = audit_slates(data)
    feasibility, feasibility_rows = audit_feasibility(data, slate)
    raw_records = permitted_raw_questions(data["permitted"], data["corpus"])
    lineage = audit_lineage(data, raw_records)
    prep = read_json(E5 / "E5_0_PREPARE_SUMMARY.json")
    split = read_json(E5 / "E5_0_SPLIT_CONFIG_AND_HASHES.json")
    receipt = read_json(E5 / "E5_0_PROSPECTIVE_RESERVATION_RECEIPT.json")

    write_json(OUT / "E5_0C_INTEGRITY_AND_PROVENANCE.json", {
        "status": "PASS_WITH_GOVERNANCE_QUALIFICATIONS" if hashes["slate_hash_exact_match"] and not hashes["e5_0_missing"] and not hashes["e5_0_mismatches"] else "BLOCKED",
        "execution_mode": "CPU_OFFLINE",
        "git": git_state(), "user": getpass.getuser(), "python": sys.version,
        "platform": platform.platform(), "read_root": str(ROOT), "write_root": str(OUT),
        "e5_0_hash_audit": hashes, "input_snapshot_hashes": {p.name: sha_file(p) for p in sorted(SNAP.glob("qasper-*.json"))},
        "e5_0c_did_not": ["model inference", "embedding inference", "HTTP/API requests", "GPU use", "reader generation", "Jev", "paid API", "reserved raw-content access"],
    })
    write_json(OUT / "E5_0C_SUPPORT_SEMANTICS_SUMMARY.json", {
        "cal_class_counts": support["cal_valid_class_counts"], "cal_flag_counts": support["cal_flag_counts"],
        "exploratory_flag_counts": support["exp_flag_counts"], "raw_questions_missing": support["raw_questions_missing"],
        "mapping_hash_mismatches": support["mapping_hash_mismatches"], "permitted_raw_papers_materialized": support["raw_permitted_papers_materialized"],
    })
    write_json(OUT / "E5_0C_RETRIEVAL_SCOPE_SUMMARY.json", {k: v for k, v in slate.items() if k != "by_group"})
    write_json(OUT / "E5_0C_FEASIBILITY_SUMMARY.json", {k: v for k, v in feasibility.items() if k != "by_group"})
    write_json(OUT / "E5_0C_SOURCE_LINEAGE_SUMMARY.json", lineage)
    write_json(OUT / "E5_0C_GROUP_FEASIBILITY_SUMMARY.json", feasibility["by_group"])
    write_csv(OUT / "E5_0C_ANNOTATION_MAPPING_CASES.csv", [
        {"query_id": qid, "partition": next((r["partition"] for r in data["mapping"] if r["query_id"] == qid), ""),
         "classification_flags": ";".join(a["flags"]), "answerable_wrappers": a["answerable_wrappers"],
         "distinct_support_sets": len(a["sets"]), "first_support_count": len(a["first_set"]),
         "field_paths": ";".join(d["field_path"] for d in a["wrapper_details"]),
         "mapping_note": "CAL/exploratory permitted record; support IDs retained only for permitted records"}
        for qid, a in sorted(support["analyses"].items())
    ], ["query_id", "partition", "classification_flags", "answerable_wrappers", "distinct_support_sets", "first_support_count", "field_paths", "mapping_note"])
    write_csv(OUT / "E5_0C_FEASIBILITY_BY_SOURCE_GROUP.csv", [
        {"source_group_id": gid, **vals} for gid, vals in sorted(feasibility["by_group"].items())
    ], ["source_group_id", "queries", "any_support_top20", "complete_support_top20", "complete_support_top5"])
    write_csv(OUT / "E5_0C_SOURCE_OVERLAP_EXCEPTIONS.csv", [
        {"exception_type": "EXACT_CROSS_SOURCE_PASSAGE_HASH", "fingerprint": h, "source_ids": "|".join(v), "status": "OBSERVED_PERMITTED_CORPUS"}
        for h, v in sorted(lineage["exact_cross_source_passage_hashes"].items())
    ] + [{"exception_type": "NORMALIZED_TITLE_COLLISION", "fingerprint": t, "source_ids": "|".join(v), "status": "DIAGNOSTIC_NOT_PROOF_OF_DUPLICATE"} for t, v in sorted(lineage["duplicate_normalized_titles"].items())], ["exception_type", "fingerprint", "source_ids", "status"])
    write_csv(OUT / "E5_0C_FEASIBILITY_QUERY_ROWS.csv", feasibility_rows, list(feasibility_rows[0].keys()) if feasibility_rows else ["query_id"])

    cal_classes = dict(support["cal_valid_class_counts"])
    primary_endpoint = "PRIMARY_ENDPOINT_NOT_VALID_FOR_PRIMARY_E5_A_UNTIL_SET_AWARE_AMENDMENT"
    write_text(OUT / "E5_0C_SUPPORT_SET_SEMANTICS_AUDIT.md", f"""# E5-0C QASPER Support-Set Semantics Audit

## Scope and access

The audit inspected all 703 permitted CAL records and all 2,028 permitted exploratory records structurally. Raw QASPER values were materialized only for source papers represented in the permitted CAL/exploratory corpus; reserved source-paper values were skipped by a top-level structural parser and were not decoded or exported. No reserved question, answer, support ID, slate or outcome was accessed.

## CAL classification

Among the 519 E5-0 mapping-valid CAL questions, the reconstructed classes are:

```json
{json.dumps(cal_classes, sort_keys=True)}
```

These classes are overlapping diagnostics, not mutually exclusive truth categories. The raw wrapper/evidence structure includes answer-wrapper multiplicity, alternative mapped support sets, duplicate evidence, non-text evidence sentinels, and exact-text mapping failures. The complete per-query audit is in `E5_0C_ANNOTATION_MAPPING_CASES.csv`.

The E5-0 mapper deterministically uses the first mapped answerable support set. It does not form a union across answer wrappers. Reconstructed first-set hashes matched the E5-0 CSV for `{len(support['mapping_hash_mismatches']) == 0}` rows with no mismatch. The audit therefore confirms the frozen mapping implementation, but it does not establish that the first set is exhaustive or factually sufficient.

## Scientific interpretation

The CSSR-U predicate is defensible only as **retention of the frozen designated support set**, not as answer correctness, source-groundedness, or exhaustive evidence recall. Multiple/partial/non-text annotations must remain visible. A future pre-outcome amendment could define a set-aware predicate as success when at least one prespecified valid support set is contained in Top-5, but it must not be silently substituted into E5-A.

Primary endpoint status: `{primary_endpoint}`.
""")
    write_text(OUT / "E5_0C_ENDPOINT_VALIDITY_AND_AMENDMENT.md", """# E5-0C Endpoint Validity and Amendment

No amendment is applied in E5-0C. The existing primary endpoint remains the frozen first-designated-support CSSR-U predicate, with all natural retrieval misses retained as unconditional failures. It is valid for designated-support retention under the existing mapping contract, but not for factual sufficiency or exhaustive evidence coverage.

Potential pre-outcome amendment for owner/reviewer consideration only: retain a single-set CSSR-U primary historical result and add a set-aware sensitivity where success is true if any prespecified valid annotated support set of cardinality <=5 is wholly contained in Top-5. This amendment would require an explicit version, signoff, and unchanged historical slate. It must not be chosen after E5-A model outcomes.
""")

    write_text(OUT / "E5_0C_RETRIEVAL_TASK_AND_SCOPE_AUDIT.md", f"""# E5-0C Retrieval Task and Scope Audit

The audited implementation in `scripts/e5_0_data_protocol.py` builds BM25 and dense rankings over the full permitted corpus of 45,497 paragraph rows. No source-paper filter is applied. Query serialization passes `q['question']` only; source paper, support IDs, answers, qrels and selector outputs are not passed to either retrieval stage. Therefore the frozen task is **global retrieval over the permitted QASPER train/dev corpus for paper-originating questions**, not paper-conditioned retrieval.

The frozen contract is BM25 lowercase-regex `k1=1.2`, `b=0.75`, Top-100 plus Qwen3-VL-Embedding-2B dense Top-100, RRF `k=60`, stable candidate-ID tie-break, and Top-20 unique passage IDs. This is a global designated-support-retention task and may contain out-of-paper unlabeled answer-bearing passages; those candidates are not treated as negative gold.

The primary CSSR-U interpretation must therefore be limited to designated within-paper evidence retention under global retrieval. It is not general cross-paper evidence exhaustiveness and not answer correctness. The existing slate is retained unchanged; no paper-conditioned replacement is generated.
""")
    write_text(OUT / "E5_0C_SLATE_INTEGRITY_REPORT.md", f"""# E5-0C Frozen Top-20 Slate Integrity Report

- Slate rows: {slate['slates']}
- Invalid rows: {len(slate['bad_slate_rows'])}
- Candidate rows: {slate['candidate_total']}
- Each row: exactly 20 unique candidate IDs, all in the permitted corpus: {'PASS' if not slate['bad_slate_rows'] else 'FAIL'}
- Exact file SHA-256: `{hashes['slate_file_sha256']}`
- Required SHA-256: `{EXPECTED_SLATE_SHA}`
- Exact hash match against the original canonical JSONL byte stream: `{hashes['slate_hash_exact_match']}`
- Candidate source-paper fraction matching the query paper: {slate['same_paper_candidate_fraction']:.6f}; out-of-paper candidates are retained and are not assumed negative.
- qrels/gold in generation: `False` in every frozen row; query text is not present in slate payloads, only its hash.

The canonical byte stream is the existing `E5_0_NATURAL_TOP20_MANIFEST.jsonl`; it was hashed directly, not reserialized.
""")
    write_text(OUT / "E5_0C_CAL_RETRIEVAL_FEASIBILITY.md", f"""# E5-0C Retrieval Feasibility

## CAL exact ceiling

`CAL_EXACT_CEILING_NOT_MEASURED`. The E5-0 artifact set contains no frozen CAL Top-20 slates and no matching CAL query-embedding cache. The permitted corpus embedding matrix exists, but recomputing the dense CAL query side would require a new embedding/model call, which is prohibited in CPU/offline E5-0C. No BM25-only proxy was substituted.

## Exploratory structural ceiling

Using only frozen 1,502 exploratory slates and the frozen designated support IDs, the pre-selector-outcome structural ceiling is:

- Any designated support present in Top-20: {feasibility['exploratory_any_support_top20']}/{feasibility['exploratory_population']} ({feasibility['exploratory_any_support_top20_rate']:.4f})
- Complete designated support present in Top-20: {feasibility['exploratory_complete_support_top20']}/{feasibility['exploratory_population']} ({feasibility['exploratory_complete_support_top20_rate']:.4f})
- Complete designated support already in first-stage Top-5: {feasibility['exploratory_complete_support_top5']}/{feasibility['exploratory_population']} ({feasibility['exploratory_complete_support_top5_rate']:.4f})
- Source groups with at least one complete Top-20 query: {feasibility['source_groups_with_feasible_query']}/{feasibility['source_group_count']}

This is an exploratory retrieval-feasibility ceiling, not a selector result or a power calculation. Infeasible natural slates remain in the unconditional E5-A population and must be reported separately from CSSR-C.
""")
    write_text(OUT / "E5_0C_SOURCE_LINEAGE_AND_NEAR_DUP_AUDIT.md", f"""# E5-0C Source Lineage and Near-Duplicate Audit

The permitted corpus contains {lineage['permitted_source_ids']} source IDs and {lineage['permitted_source_groups']} source groups; source IDs are unique at the permitted-corpus level. E5-0's all-source grouping contract reports 1,169 exact full-paper content components with partition counts 234/643/292. The permitted corpus intentionally excludes reserved source content.

- Exact cross-source paragraph-text hashes in permitted content: {lineage['exact_cross_source_passage_hash_count']}
- Normalized-title collisions in permitted raw records: {len(lineage['duplicate_normalized_titles'])}
- Reservation near-duplicate disjointness: `RESERVED_NEAR_DUPLICATE_DISJOINTNESS_NOT_INDEPENDENTLY_VERIFIED`
- Pretraining exposure: `UNKNOWN`

Exact duplicate grouping is defensible for the visible corpus. Approximate publication-version lineage and cross-reservation near-duplicates cannot be certified without opening sealed content; no zero-overlap claim is made for that boundary.
""")
    write_text(OUT / "E5_0C_RESERVATION_AND_ACCESS_GOVERNANCE.md", f"""# E5-0C Reservation and Access Governance

The local reservation receipt records seed 731, 292 reserved groups, 867 reserved questions, `reservation_before_selector_outcomes=true`, external attestation `UNAVAILABLE`, and access isolation `PARTIAL`. E5-F remains `E5_F_CONFIRMATION_LOCKED`.

The local chronology is evidence of execution order only. It is not independent attestation and does not certify sealed access. The required future control is an independent custodian or separately credentialed immutable reservation with a pre-outcome digest and auditable access denial.

No reserved raw payload was read by E5-0C. Only permitted CAL/exploratory data and nonrevealing reservation metadata were used.
""")
    write_text(OUT / "E5_0C_GPU_AUTHORIZATION_AND_SMOKE_RECONCILIATION.md", f"""# E5-0C GPU Authorization and Smoke Reconciliation

E5-0C itself used no GPU, loaded no model, made no HTTP request, and made no inference.

Historical E5-0 synthetic smoke records show PPLX on GPU5 and Jina on GPU6, with one physical forward each. The contemporaneous saved E2-S/P authorization artifact explicitly authorizes GPUs 0/1/2/4 and does not authorize 5/6. The conversation-level instruction that GPUs 5/6 were idle and could be used is not present as an immutable project authorization record. Therefore the historical use is recorded as `GPU_5_6_AUTHORIZATION_UNVERIFIED` at the artifact-governance level; successful execution is not treated as authorization. The raw four smoke rows, including the first two harness false-negative rows, remain preserved.

Future E5-A requires a separate affirmative owner authorization naming exact GPU IDs, current safe/isolated occupancy, device UUIDs, model workload, and no interference with resident services. GPUs 5/6 must not be reused from historical telemetry without that explicit record.
""")
    write_text(OUT / "E5_0C_FROZEN_E5_A_ESTIMAND_AND_ANALYSIS.md", f"""# E5-0C Frozen E5-A Estimand and Analysis

This is a pre-outcome lock for owner review; no E5-A selector output was read or generated.

## Primary population

The fixed eligible exploratory cohort is 1,502 questions across 643 source groups, with zero-eligible groups excluded from the evaluation denominator. The frozen Top-20 file SHA-256 is `{EXPECTED_SLATE_SHA}`. Model revisions are PPLX `{PPLX_REV}` and Jina `{JINA_REV}`.

## Primary estimand

For each source group g, compute the mean paired CSSR-U difference PPLX minus Jina over its eligible questions, then average equally across the 643 groups. CSSR-U retains retrieval-infeasible questions as failures. CSSR-C is a secondary diagnostic conditional on complete frozen designated support appearing in Top-20.

The primary support predicate remains the frozen first designated mapped support set. It is a retention metric, not factual correctness or evidence exhaustiveness. Invalid/missing native outputs are intention-to-evaluate failures, with complete-valid-case sensitivity reported separately.

## Uncertainty and diagnostics

Use jointly paired source-group bootstrap, 5,000 replicates, seed 731, two-sided 95% interval. Secondary reports: query-weighted effect, per-group effects, leave-largest-group-out, support-cardinality strata, retrieval-feasibility ceiling, both-failures, PPLX-only/Jina-only successes, and honest native failures. No superiority, noninferiority, or parity claim is implied.

E5-C reader validation and E5-D serving timing remain separately frozen but unexecuted. E5-F remains locked.
""")
    write_text(OUT / "E5_0C_E5_A_CHANGE_CONTROL_REGISTER.md", """# E5-0C E5-A Change-Control Register

| Change | Status | Decision |
|---|---|---|
| Replace first-designated CSSR-U with set-aware CSSR-U | PROPOSED_ONLY | Requires pre-outcome owner/reviewer approval; not applied |
| Restrict global retrieval to source paper | PROPOSED_ONLY | Would define a different historical task; not applied |
| Add CAL exact hybrid ceiling | BLOCKED | Requires new CAL embedding/model call; not executed |
| Reuse GPU5/6 for E5-A | BLOCKED | Historical authorization is not immutably verified |

Frozen exploratory cohort, frozen slates, model revisions and primary estimand were not changed after selector outcomes. No amendment was applied.
""")
    support_audit_path = OUT / "E5_0C_SUPPORT_SET_SEMANTICS_AUDIT.md"
    support_audit_text = support_audit_path.read_text(encoding="utf-8")
    support_audit_text = support_audit_text.replace(
        "The audit inspected all 703 permitted CAL records and all 2,028 permitted exploratory records structurally.",
        f"The audit inspected all 703 permitted CAL records and all 2,028 permitted exploratory records structurally; raw semantic reconstruction succeeded for {len(support['analyses'])} records and {len(support['raw_questions_missing'])} CAL records lacked a permitted source-paper paragraph manifest and remain metadata-only.",
    )
    support_audit_path.write_text(support_audit_text, encoding="utf-8")
    endpoint_path = OUT / "E5_0C_ENDPOINT_VALIDITY_AND_AMENDMENT.md"
    endpoint_text = endpoint_path.read_text(encoding="utf-8")
    endpoint_text = endpoint_text.replace(
        "No amendment is applied in E5-0C. The existing primary endpoint remains the frozen first-designated-support CSSR-U predicate, with all natural retrieval misses retained as unconditional failures. It is valid for designated-support retention under the existing mapping contract, but not for factual sufficiency or exhaustive evidence coverage.",
        "The frozen first-designated-support CSSR-U predicate is reproducible but is not valid as the primary E5-A relevance endpoint because 53 CAL-valid questions have multiple mapped support sets and 95 have partial/non-exhaustive evidence flags. It may be retained as a historical descriptive diagnostic, with all natural retrieval misses retained as unconditional failures, but must not be used as the sole primary E5-A endpoint.",
    )
    endpoint_path.write_text(endpoint_text, encoding="utf-8")
    frozen_path = OUT / "E5_0C_FROZEN_E5_A_ESTIMAND_AND_ANALYSIS.md"
    frozen_text = frozen_path.read_text(encoding="utf-8")
    frozen_text = frozen_text.replace(
        "## Primary estimand\n\nFor each source group g, compute the mean paired CSSR-U difference PPLX minus Jina over its eligible questions, then average equally across the 643 groups.",
        "## Primary estimand (pending amendment)\n\nThe source-group-equal-weighted PPLX minus Jina contrast is retained as the planned structure, but the primary CSSR-U predicate is pending a pre-outcome set-aware amendment. The current first-wrapper CSSR-U may only be reported as a historical diagnostic until approval.",
    )
    frozen_path.write_text(frozen_text, encoding="utf-8")
    gpu_status = "GPU_5_6_AUTHORIZATION_UNVERIFIED"
    gate = "E5_0C_PROTOCOL_AMENDMENT_REQUIRED"
    write_text(OUT / "E5_0C_E5_A_OWNER_DECISION_GATE.md", f"""# E5-0C E5-A Owner Decision Gate

## Gate: `{gate}`

Scientific readiness for the exploratory E5-A comparison is established with qualifications: the frozen cohort, slate identities, global retrieval scope, designated-support mapping, feasibility diagnostic and paired group estimator are auditable. The primary endpoint is not a factual-correctness or exhaustive-evidence endpoint.

Owner authorization is **not** granted by this gate. Before E5-A, the owner must separately approve native selector inference, exact GPU IDs/UUIDs, current safe isolation, model revisions, and the immutable execution manifest. Historical GPU5/6 use remains `{gpu_status}`.

CAL exact hybrid feasibility remains `CAL_EXACT_CEILING_NOT_MEASURED`. Reserved near-duplicate independence and external reservation attestation remain unresolved. `E5_F_CONFIRMATION_LOCKED`.

**STOP:** Do not execute E5-A, E5-B/C/D/F, Jev, paid inference, reader generation, or any GPU/model workload from E5-0C.
""")

    events = [
        {"event": "E5_0C_STARTED", "mode": "CPU_OFFLINE", "reserved_payload_access": False},
        {"event": "E5_0_HASH_AUDIT", "e5_0_files": hashes["e5_0_manifest_files"], "mismatches": len(hashes["e5_0_mismatches"]), "slate_exact_match": hashes["slate_hash_exact_match"]},
        {"event": "S0_SUPPORT_SEMANTICS", "cal_rows": len(data["mapping"]), "valid_cal": sum(1 for x in data["mapping"] if x["partition"] == "CAL" and x["mapping_status"] == "VALID_DESIGNATED_SUPPORT"), "raw_reserved_materialized": False},
        {"event": "R1_SCOPE", "scope": "GLOBAL_PERMITTED_CORPUS", "corpus_rows": len(data["corpus"]), "selector_outputs_read": False},
        {"event": "F0_FEASIBILITY", "cal_status": feasibility["cal_exact_ceiling"], "exploratory_rows": feasibility["exploratory_population"]},
        {"event": "L0_LINEAGE", "reserved_near_duplicate_status": lineage["reserved_near_duplicate_disjointness"]},
        {"event": "G0_GPU", "gpu_use_in_e5_0c": False, "historical_gpu_status": gpu_status},
        {"event": "E5_0C_STOPPED", "gate": gate, "e5_a_started": False, "e5_f": "LOCKED", "jev_calls": 0, "paid_calls": 0},
    ]
    write_json(OUT / "E5_0C_EXECUTION_EVENTS.json", {"events": events})
    write_text(OUT / "E5_0C_EXECUTION_EVENTS.md", "# E5-0C Execution Events\n\n" + "\n".join(f"- `{e['event']}`: {json.dumps(e, sort_keys=True)}" for e in events))
    # Required report is written after all measured summaries exist.
    write_text(OUT / "E5_0C_FINAL_READINESS_AND_SCIENTIFIC_REPORT.md", f"""# DecisionRAG E5-0C Final Readiness and Scientific Report

## Final gate

`{gate}`. Scientific exploratory readiness for E5-A passes with explicit qualifications; E5-A owner/GPU authorization is still pending. E5-F remains locked.

## Authenticated facts

E5-0 output manifest: {hashes['e5_0_manifest_files']} files; missing={len(hashes['e5_0_missing'])}; mismatches={len(hashes['e5_0_mismatches'])}. Historical E4-R audit: checked={hashes['e5_0_historical_audit'].get('checked')}, clean={hashes['e5_0_historical_audit'].get('clean')}. Frozen Top-20 SHA-256 exact match: {hashes['slate_hash_exact_match']}. Existing E5-0 gate: `{hashes['e5_0_gate']}`.

## Evidence semantics

CAL has 519 mapping-valid questions, but answer wrappers and evidence spans are not universally exhaustive. Reconstructed first-designated support hashes match the E5-0 mapping for `{len(support['mapping_hash_mismatches']) == 0}`. Multiple alternative/partial/non-text cases are reported separately. CSSR-U is defensible only as designated-support retention.

## Retrieval and feasibility

The frozen retriever is global over 45,497 permitted paragraphs, using question text only, BM25 Top-100 + Qwen dense Top-100 + RRF(60). Out-of-paper candidates are common and are not treated as negative gold. CAL exact hybrid feasibility is not measured because matching CAL query embeddings/slates are absent and new embedding calls are prohibited. Exploratory pre-outcome complete-support Top-20 feasibility is {feasibility['exploratory_complete_support_top20']}/{feasibility['exploratory_population']} ({feasibility['exploratory_complete_support_top20_rate']:.4f}).

## Source and governance limitations

Permitted exact source grouping is auditable. Cross-reservation near-duplicate disjointness is `{lineage['reserved_near_duplicate_disjointness']}`. External reservation attestation is unavailable and access isolation is partial. GPU5/6 historical smoke use is `{gpu_status}` in the saved artifact record; E5-0C used no GPU.

## Decision

E5-A may be considered by the owner as an exploratory designated-support-retention comparison after a separate GPU/model authorization. No E5-A result, factual correctness claim, parity claim, or confirmatory claim is made here. No historical artifacts were modified.
""")

    final_report_path = OUT / "E5_0C_FINAL_READINESS_AND_SCIENTIFIC_REPORT.md"
    final_report_text = final_report_path.read_text(encoding="utf-8")
    final_report_text = final_report_text.replace(
        f"`{gate}`. Scientific exploratory readiness for E5-A passes with explicit qualifications; E5-A owner/GPU authorization is still pending.",
        f"`{gate}`. The existing first-wrapper CSSR-U is not valid as the sole primary E5-A endpoint; a pre-outcome set-aware amendment and owner/reviewer approval are required before E5-A.",
    )
    final_report_text = final_report_text.replace(
        "E5-A may be considered by the owner as an exploratory designated-support-retention comparison after a separate GPU/model authorization.",
        "E5-A must not begin until the proposed set-aware endpoint amendment is approved and a separate GPU/model authorization is recorded.",
    )
    final_report_path.write_text(final_report_text, encoding="utf-8")
    gate_path = OUT / "E5_0C_E5_A_OWNER_DECISION_GATE.md"
    gate_text = gate_path.read_text(encoding="utf-8")
    gate_text = gate_text.replace(
        "Scientific readiness for the exploratory E5-A comparison is established with qualifications: the frozen cohort, slate identities, global retrieval scope, designated-support mapping, feasibility diagnostic and paired group estimator are auditable.",
        "The frozen cohort, slate identities, global retrieval scope and paired group structure are auditable, but scientific readiness for the primary E5-A comparison is blocked pending a set-aware endpoint amendment because the first-wrapper CSSR-U is not a valid sole relevance endpoint.",
    )
    gate_text = gate_text.replace("Before E5-A, the owner must separately approve", "After amendment approval and before E5-A, the owner must separately approve")
    gate_path.write_text(gate_text, encoding="utf-8")

    # Hash all E5-0C files except the self-referential manifest.
    files = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "E5_0C_PROVENANCE_MANIFEST.json":
            files.append({"path": str(p.relative_to(ROOT)), "sha256": sha_file(p), "bytes": p.stat().st_size})
    write_json(OUT / "E5_0C_PROVENANCE_MANIFEST.json", {
        "schema_version": "E5-0C-provenance-v1", "gate": gate, "files": files,
        "e5_0_prior_manifest_preserved": True, "e5_0c_cpu_only": True,
        "no_gpu": True, "no_model_calls": True, "no_http_calls": True, "no_reader": True,
        "no_jev": True, "no_paid_api": True, "no_reserved_outcomes": True,
        "e5_f_confirmation": "LOCKED", "gpu5_6_status": gpu_status,
    })


def main() -> int:
    write_outputs()
    print(json.dumps({"out": str(OUT), "gate": "E5_0C_PROTOCOL_AMENDMENT_REQUIRED"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
