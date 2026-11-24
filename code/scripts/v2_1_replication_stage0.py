#!/usr/bin/env python3
"""CPU-only Stage-0 preflight for the v2.1 controlled replication.

This deliberately stops before model scoring.  The execution contract requires
an existing, exact BM25 Top-20 slate.  The historical v1.1 traces available in
this repository are hybrid BM25+dense RRF Top-20 traces, so this script records
that mismatch and prepares an auditable blocker rather than creating a new
retrieval population.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v2_1_replication"
STAGE0 = OUT / "stage0_preflight"
PPLX_REV = "14c9d90137962d300f3de95633f7b6535e80c5ce"
JINA_REV = "e8a93f33f0b22108f8c2364f8484ce3422552fbc"
PPLX_PATH = Path(f"<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--perplexity-ai--pplx-decider-v1.1-27b/snapshots/{PPLX_REV}")
MAX_CONTEXT = 8192
MARGIN = 128
SEEDS = {"bootstrap": 731, "shuffle_1": 20261020, "shuffle_2": 20261021, "cohort_enterprise": 20261020, "cohort_musique": 20261021, "determinism": 20261022}

ENTERPRISE_QUESTIONS = ROOT / "results/preflight/v11_remediation/ENTERPRISE_QUESTION_MANIFEST_V11R1.jsonl"
ENTERPRISE_GOLD = ROOT / "results/preflight/v11/ENTERPRISE_GOLD_SIDECAR.jsonl"
ENTERPRISE_SLATES = ROOT / "results/non_jev/v11/track_b/V11_TRACK_B_RETRIEVAL_SLATES.jsonl"
ENTERPRISE_CHUNKS = ROOT / "results/non_jev/v11/track_b/V11_TRACK_B_CHUNK_MANIFEST.jsonl"
ENTERPRISE_ZIP = ROOT / "data/raw/vendor/enterprise_rag_bench/v1.0.0/all_documents.zip"
MUSIQUE_DEV = ROOT / "data/raw/vendor/musique/musique_ans_v1.0_dev.jsonl"
MUSIQUE_IDS = ROOT / "results/preflight/v11/NEW_TEST_IDS.jsonl"
MULTI_SLATES = ROOT / "results/non_jev/v11/track_a/V11_TRACK_A_RETRIEVAL_SLATES.jsonl"
CORPUS = ROOT / "data/corpus/passages.jsonl"
E11_CONFIG = ROOT / "results/non_jev/v2/e11_v1_0/01_budget_policy/E11_A_BUDGET_POLICY_CONFIG.json"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_json(value) -> str:
    return sha_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_once(path: Path, value, *, raw: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    if raw:
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl_once(path: Path, rows: list[dict]) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def quantile(values: list[int], q: float):
    if not values:
        return None
    arr = sorted(values)
    pos = (len(arr) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(arr) - 1)
    return arr[lo] if lo == hi else arr[lo] + (arr[hi] - arr[lo]) * (pos - lo)


def source_record_id(query_id: str) -> str:
    return query_id.split(":", 1)[1] if ":" in query_id else query_id


def audit_enterprise() -> dict:
    questions = read_jsonl(ENTERPRISE_QUESTIONS)
    gold = read_jsonl(ENTERPRISE_GOLD)
    slates = read_jsonl(ENTERPRISE_SLATES)
    chunks = read_jsonl(ENTERPRISE_CHUNKS)
    qids = {x["question_id"] for x in questions}
    slate_qids = {x["query_id"] for x in slates}
    chunk_ids = {x["internal_doc_id"] for x in chunks}
    candidate_ids = {cid for row in slates for cid in row.get("candidate_ids", [])}
    bm25_top100_present = all(bool(x.get("bm25_top100")) for x in slates)
    derived_bm25_prefix = []
    for row in slates:
        derived = [x.get("doc_id") for x in row.get("bm25_top100", [])[:20]]
        derived_bm25_prefix.append({"query_id": row["query_id"], "candidate_ids": derived, "n": len(derived)})
    return {
        "dataset": "EnterpriseRAG-Bench",
        "source_files": {str(p.relative_to(ROOT)): sha_file(p) for p in (ENTERPRISE_QUESTIONS, ENTERPRISE_GOLD, ENTERPRISE_SLATES, ENTERPRISE_CHUNKS, ENTERPRISE_ZIP)},
        "questions": len(questions),
        "unique_question_ids": len(qids),
        "gold_rows": len(gold),
        "gold_status_counts": dict(Counter(x.get("cssr_denominator_status") for x in questions)),
        "potentially_labeled_questions": sum(x.get("cssr_denominator_status") == "APPLICABLE" for x in questions),
        "qst_0413_status": next((x.get("gold_identity_status") for x in questions if x.get("question_id") == "qst_0413"), None),
        "frozen_prior_slate_rows": len(slates),
        "frozen_prior_slate_unique_ids": len(slate_qids),
        "prior_slate_candidate_count": dict(Counter(len(x.get("candidate_ids", [])) for x in slates)),
        "candidate_ids_resolve_to_chunk_manifest": sum(cid in chunk_ids for cid in candidate_ids),
        "candidate_ids_total": len(candidate_ids),
        "bm25_top100_present_in_prior_trace": bm25_top100_present,
        "prior_candidate_order_contract": "not recorded as BM25-only; prior V11 trace is hybrid RRF Top-20",
        "derived_bm25_prefix_rows_for_diagnostic_only": len(derived_bm25_prefix),
        "exact_required_bm25_top20_slate_available": False,
        "exact_candidate_text_adapter_available": False,
        "blocker_code": "ENTERPRISE_EXACT_BM25_TOP20_INPUT_MISSING",
        "labels_not_used_for_inference": True,
    }


def audit_musique() -> tuple[dict, list[dict], dict[str, dict]]:
    dev = read_jsonl(MUSIQUE_DEV)
    selected = [x for x in read_jsonl(MUSIQUE_IDS) if x.get("dataset") == "musique"]
    selected_ids = {x["source_record_id"] for x in selected}
    dev_by_id = {x["id"]: x for x in dev}
    slates = [x for x in read_jsonl(MULTI_SLATES) if x.get("dataset") == "musique" and source_record_id(x["query_id"]) in selected_ids]
    corpus = {x["doc_id"]: x for x in read_jsonl(CORPUS)}
    missing_records = sorted(selected_ids - set(dev_by_id))
    missing_candidates = sorted({cid for row in slates for cid in row.get("candidate_ids", []) if cid not in corpus})
    support_counts = Counter()
    for rid in selected_ids:
        rec = dev_by_id.get(rid)
        if rec:
            support_counts[sum(bool(p.get("is_supporting")) for p in rec.get("paragraphs", []))] += 1
    return ({
        "dataset": "MuSiQue",
        "source_files": {str(p.relative_to(ROOT)): sha_file(p) for p in (MUSIQUE_DEV, MUSIQUE_IDS, MULTI_SLATES, CORPUS)},
        "official_dev_rows": len(dev),
        "selected_population_rows": len(selected),
        "selected_population_unique_source_ids": len(selected_ids),
        "selected_records_resolve": len(selected_ids) - len(missing_records),
        "selected_record_missing_ids": missing_records,
        "selected_answerable_counts": dict(Counter(dev_by_id[x]["answerable"] for x in selected_ids if x in dev_by_id)),
        "support_cardinality_counts": dict(sorted(support_counts.items())),
        "frozen_prior_slate_rows": len(slates),
        "prior_slate_unique_query_ids": len({x["query_id"] for x in slates}),
        "prior_slate_candidate_count": dict(Counter(len(x.get("candidate_ids", [])) for x in slates)),
        "prior_candidate_order_contract": dict(Counter(x.get("candidate_order") for x in slates)),
        "candidate_text_resolve_count": len({cid for row in slates for cid in row.get("candidate_ids", []) if cid in corpus}),
        "candidate_text_missing_count": len(missing_candidates),
        "candidate_text_missing_ids": missing_candidates[:100],
        "exact_required_bm25_top20_slate_available": False,
        "exact_bm25_top100_or_order_available": False,
        "exact_candidate_text_adapter_available": not missing_candidates,
        "blocker_code": "MUSIQUE_EXACT_BM25_TOP20_INPUT_MISSING",
        "labels_not_used_for_inference": True,
    }, slates, corpus)


def make_pplx_row(question: str, docs: list[dict], codes: list[str]) -> dict:
    criteria = {codes[i]: f"Candidate ID {d['candidate_id']}. Passage: {d['content']}" for i, d in enumerate(docs)}
    return {"state": f"Query: {question}\nEvaluate the supplied candidate passages using the option descriptions.", "question": {"type": "choice", "instructions": f"Select the candidate passage that best answers the query: {question}", "criteria": criteria}}


def label_free_length_scan(musique_slates: list[dict], corpus: dict[str, dict]) -> dict:
    result = {"status": "NOT_RUN", "reason": None, "dataset": "MuSiQue", "scopes": {}}
    try:
        sys.path.insert(0, str(PPLX_PATH / "source" / "src"))
        from autojev.model import decision_messages
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(str(PPLX_PATH), local_files_only=True)
        codes20 = [f"candidate_{i:03d}" for i in range(20)]
        codes10 = [f"candidate_{i:03d}" for i in range(10)]

        def count(question: str, docs: list[dict], codes: list[str]) -> int:
            messages = decision_messages(make_pplx_row(question, docs, codes), codes)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            encoded = processor(text=[text], images=None, padding=True, return_tensors="pt")
            return int(encoded["attention_mask"].sum())

        values = {"single-call": [], "window A": [], "window B": []}
        manifest_rows = []
        for row in musique_slates:
            docs = [{"candidate_id": cid, "content": corpus[cid]["full_text"]} for cid in row["candidate_ids"] if cid in corpus]
            if len(docs) != 20:
                continue
            question = row["question"]
            lengths = {"single-call": count(question, docs, codes20), "window A": count(question, docs[:10], codes10), "window B": count(question, docs[10:], codes10)}
            for scope, value in lengths.items():
                values[scope].append(value)
            manifest_rows.append({"query_id": row["query_id"], "candidate_ids": row["candidate_ids"], "lengths": lengths, "input_sha256": sha_json({"question": question, "candidate_ids": row["candidate_ids"], "lengths": lengths})})
        result.update({"status": "COMPLETE_DIAGNOSTIC_ONLY", "context_limit": MAX_CONTEXT, "safety_margin": MARGIN, "tokenizer_revision": PPLX_REV, "n_queries": len(manifest_rows), "scopes": {scope: {"n": len(vals), "share_over_limit": sum(v > MAX_CONTEXT for v in vals) / len(vals) if vals else None, "q50": quantile(vals, .5), "q90": quantile(vals, .9), "q99": quantile(vals, .99), "max": max(vals) if vals else None} for scope, vals in values.items()}, "input_manifest_rows": manifest_rows})
    except Exception as exc:
        result.update({"status": "BLOCKED_TOKENIZER_PREFLIGHT", "reason": f"{type(exc).__name__}: {exc}"})
    return result


def main() -> int:
    if STAGE0.exists() and (STAGE0 / "STATUS.md").exists():
        print(json.dumps({"status": "ALREADY_WRITTEN", "output": str(STAGE0.relative_to(ROOT))}, indent=2))
        return 0
    enterprise = audit_enterprise()
    musique, musique_slates, corpus = audit_musique()
    length_scan = label_free_length_scan(musique_slates, corpus)
    source_hashes = {}
    for path in (ENTERPRISE_QUESTIONS, ENTERPRISE_GOLD, ENTERPRISE_SLATES, ENTERPRISE_CHUNKS, ENTERPRISE_ZIP, MUSIQUE_DEV, MUSIQUE_IDS, MULTI_SLATES, CORPUS, E11_CONFIG):
        source_hashes[str(path.relative_to(ROOT))] = sha_file(path)

    protocol = {
        "protocol_id": "decisionrag-v2.1-controlled-replication",
        "status": "FROZEN_PRE_OUTCOME_STAGE0_BLOCKED_BOTH_BENCHMARKS",
        "created_utc": utc(),
        "code_revision": "scripts/v2_1_replication_stage0.py",
        "models": {"pplx_decider": {"revision": PPLX_REV, "native_contract": "frozen PPLX two-stage selector"}, "jina_v3_5": {"revision": JINA_REV, "native_contract": "frozen Jina native scoring and deterministic Top-5"}},
        "populations": {"enterprise_requested": 500, "enterprise_quality_requested": 443, "musique_requested": 2167},
        "orders": ["original", "reversed", "perm_seed_20261020", "perm_seed_20261021"],
        "seeds": SEEDS,
        "candidate_contract": "existing exact BM25 Top-20 slates, 20 unique canonical IDs, frozen original order",
        "metric": "FS@5 complete annotated support set contained in selected Top-5; failed labeled requests score 0",
        "context": {"max_native_context_tokens": MAX_CONTEXT, "safety_margin_tokens": MARGIN, "budget_policy_source": str(E11_CONFIG.relative_to(ROOT))},
        "resampling_unit": "query; no qualifying pre-existing group identifier was found in the admissible frozen manifests",
        "label_blind_inference": True,
        "labels_joined_after_raw_trace_freeze": True,
        "no_retries_or_fill_in": True,
        "source_hashes": source_hashes,
        "gate": {"stage0": "BLOCKED", "stage1": "NOT_STARTED", "stage2": "NOT_STARTED", "stage3": "NOT_STARTED", "stage4": "CPU_ANALYSIS_NOT_APPLICABLE_WITHOUT_MODEL_OUTPUTS", "stage5": "BLOCKED", "reason": "R2_BOTH_BENCHMARKS_BLOCKED"},
        "benchmark_audits": {"enterprise": enterprise["blocker_code"], "musique": musique["blocker_code"]},
    }
    protocol_hash = sha_json(protocol)
    input_manifest = {
        "status": "REFERENCE_ONLY_NOT_ADMISSIBLE_FOR_SCORING",
        "label_blind": True,
        "source_slate_contract": "historical V11 hybrid RRF traces used only to diagnose missing exact BM25 inputs",
        "enterprise": {"query_count": enterprise["frozen_prior_slate_rows"], "candidate_ids_hashed_only": True, "exact_bm25_available": False},
        "musique": {"query_count": musique["frozen_prior_slate_rows"], "candidate_ids_hashed_only": True, "exact_bm25_available": False, "length_scan": {k: v.get("input_manifest_rows", []) for k, v in {"diagnostic": length_scan}.items()}},
        "manifest_sha256_excludes_labels": True,
    }
    write_once(STAGE0 / "PROTOCOL.json", protocol)
    write_once(STAGE0 / "PROTOCOL_SHA256.json", {"path": "stage0_preflight/PROTOCOL.json", "sha256": protocol_hash, "frozen_before_model_scoring": True})
    write_once(STAGE0 / "DATASET_INVENTORY.json", {"enterprise": enterprise, "musique": musique, "source_hashes": source_hashes})
    write_once(STAGE0 / "LABEL_FREE_LENGTH_SCAN_DIAGNOSTIC.json", length_scan)
    write_once(STAGE0 / "INPUT_MANIFEST.json", input_manifest)
    write_once(STAGE0 / "INPUT_MANIFEST_SHA256.json", {"sha256": sha_json(input_manifest), "label_fields_included": False, "admissible_for_inference": False})
    write_once(STAGE0 / "ENTERPRISE_AUDIT.json", enterprise)
    write_once(STAGE0 / "MUSIQUE_AUDIT.json", musique)
    write_once(STAGE0 / "ENTERPRISE_QUESTION_IDS.jsonl", [{"canonical_id": x["question_id"]} for x in read_jsonl(ENTERPRISE_QUESTIONS)])
    write_jsonl_once(STAGE0 / "MUSIQUE_QUESTION_IDS.jsonl", [{"canonical_id": x["query_id"], "source_record_id": x["source_record_id"]} for x in read_jsonl(MUSIQUE_IDS) if x.get("dataset") == "musique"])
    write_once(STAGE0 / "BLOCKER_ENTERPRISE.md", """# EnterpriseRAG-Bench blocker\n\nStatus: `BLOCKED` under decision rule A3-R1.\n\nThe repository contains 500 questions, 443 `APPLICABLE` identity-resolved quality rows, the official archive and a prior 500-row V11 retrieval trace. That trace stores a hybrid `BM25 top-100 + dense top-100 + RRF@60` Top-20, not an existing exact BM25-only Top-20 slate. The prior `candidate_order` is not a BM25-only order. Although BM25 top-100 lists are embedded in the trace, deriving and using their first 20 would create a new candidate population and is prohibited by the frozen replication contract. The prior chunk manifest also does not itself provide the exact selector input strings.\n\nNo model call was made. Unblocking requires an existing hashable BM25 Top-20 slate plus its exact candidate text adapter, or an owner-approved versioned protocol amendment.\n""", raw=True)
    write_once(STAGE0 / "BLOCKER_MUSIQUE.md", """# MuSiQue blocker\n\nStatus: `BLOCKED` under decision rule A3-R1.\n\nThe repository contains the 2,167 selected MuSiQue records and prior V11 hybrid RRF Top-20 traces. The prior trace records `BM25 top-100 + dense top-100 + RRF@60` output and its RRF order; it does not preserve the exact BM25 top-100/top-20 list needed to reconstruct the required BM25-only Top-20 without running a new retrieval. Running BM25 over the corpus would violate the frozen-input-only rule.\n\nAll 2,167 selected source records and their annotated support cardinalities were audited offline; no relevance outcome was used to construct this blocker. No model call was made. Unblocking requires the exact hashable BM25 Top-20 slate and candidate text manifest, or an owner-approved versioned protocol amendment.\n""", raw=True)
    deviations = """# DEVIATIONS\n\n- Stage 0 stopped under A3-R2 because neither workload has the exact required existing BM25-only Top-20 artifact.\n- The historical v1.1 hybrid RRF Top-20 traces were inspected only for availability and identity diagnostics; they were not substituted for the frozen input population.\n- No new retrieval, selector inference, timing, statistical ranking evaluation, GPU use, API request, retry or fill-in occurred.\n- The MuSiQue tokenizer length scan is marked diagnostic-only because its source slate is hybrid RRF rather than the required BM25 slate. Enterprise length scanning is not admissible because the exact candidate view adapter is absent.\n"""
    write_once(STAGE0 / "DEVIATIONS.md", deviations, raw=True)
    status = {
        "overall_status": "BLOCKED",
        "reason": "R2_BOTH_BENCHMARKS_BLOCKED",
        "model_inference_calls": 0,
        "paid_api_calls": 0,
        "jev_calls": 0,
        "gpu_workloads_started": 0,
        "stages": {f"stage{i}": {"status": "blocked" if i != 0 else "blocked", "attempted": 0, "valid": 0, "failures": {"INPUT_CONTRACT_BLOCKED": 1}} for i in range(6)},
        "next_action": "Provide exact existing BM25 Top-20 slate and candidate-text manifests, then rerun scripts/v2_1_replication_stage0.py followed by the authorized Stage 1 runner.",
        "protocol_sha256": protocol_hash,
        "source_hashes": source_hashes,
    }
    write_once(STAGE0 / "STATE.json", status)
    write_once(OUT / "STATE.json", status)
    write_once(OUT / "STATUS.md", f"""# DecisionRAG v2.1 controlled replication status\n\nOverall status: `BLOCKED` (`R2_BOTH_BENCHMARKS_BLOCKED`).\n\nStage 0 completed CPU-only inventory and froze a pre-outcome protocol receipt. Both benchmark branches lack the exact existing BM25-only Top-20 artifact required by the contract. No inference or GPU workload was started. See `stage0_preflight/BLOCKER_ENTERPRISE.md`, `stage0_preflight/BLOCKER_MUSIQUE.md`, `stage0_preflight/PROTOCOL.json`, and `stage0_preflight/DEVIATIONS.md`.\n\nProtocol SHA-256: `{protocol_hash}`\n\nResume after an approved input-contract resolution: `python3 scripts/v2_1_replication_stage0.py`\n""", raw=True)
    report = f"""# DecisionRAG v2.1 controlled replication report\n\n## Overall status\n\n`BLOCKED`: both benchmark branches stopped before model scoring under A3-R2. This is a protocol-input blocker, not a model result.\n\n## Stage 0 findings\n\n- EnterpriseRAG-Bench: 500 questions; 443 `APPLICABLE` identity-resolved quality rows; 57 non-quality rows (30 empty gold, 26 over-five gold, 1 accepted qst_0413 ambiguity exception). A prior 500-row hybrid RRF Top-20 trace exists, but the exact BM25-only Top-20 artifact required by this protocol does not.\n- MuSiQue: 2,167 selected records from the 2,417-row local dev release; support cardinalities were audited offline. A prior 2,167-row hybrid RRF Top-20 trace exists, but the exact BM25-only Top-20 artifact and preserved BM25 ordering required by this protocol do not.\n- Label-free length scan: `{length_scan.get('status')}` for the non-admissible MuSiQue reference trace; it is not a replication result.\n- Protocol was written and hashed before any model scoring call; protocol SHA-256 is `{protocol_hash}`.\n\n## Stages 1–5\n\nNo selector inference, order robustness, serving timing, FS@5 statistics, or export package was run. There are zero model calls, zero GPU workloads, zero Jev calls and zero paid API calls.\n\n## Conclusions\n\nTransferred findings: none from this replication, because no admissible model outcomes exist. Divergent findings: none measured. Unverified: all six replication questions in the specification.\n\n## Required unblock\n\nSupply the exact hashable BM25-only Top-20 candidate slates and exact candidate-text manifests for both datasets, or approve a versioned protocol amendment that explicitly permits the existing hybrid RRF slates. The historical slates must remain preserved.\n"""
    write_once(OUT / "REPLICATION_REPORT.md", report, raw=True)
    write_once(OUT / "REPLICATION_REPORT_INTERIM.md", report, raw=True)
    write_once(OUT / "DEVIATIONS.md", deviations, raw=True)
    write_once(OUT / "RAW_TRACE_SHA256_MANIFEST.json", {"status": "NO_NEW_RAW_TRACES", "raw_trace_count": 0, "historical_inputs": source_hashes})
    print(json.dumps({"overall_status": "BLOCKED", "reason": "R2_BOTH_BENCHMARKS_BLOCKED", "protocol_sha256": protocol_hash, "model_calls": 0, "output": str(OUT.relative_to(ROOT))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
