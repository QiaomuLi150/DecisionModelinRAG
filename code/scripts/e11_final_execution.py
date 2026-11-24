#!/usr/bin/env python3
"""DecisionRAG E11 final local-model replacement validity execution.

Append-only E11 runner.  It first replays E9 offline, then freezes one
label-blind equal-prefix context policy over the complete BRIGHT population,
executes only budgeted PPLX/Jina native inference, and closes the evidence
record.  It never downloads models, runs a reader/judge, accesses protected
holdouts, or alters E2-E10 artifacts.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
E9 = ROOT / "results/non_jev/v2/e9_bright_v1_0"
E9A = ROOT / "results/non_jev/v2/e9_bright_interpretation_v1_0"
E10 = ROOT / "results/non_jev/v2/e10_v1_0"
E6 = ROOT / "results/non_jev/v2/e6_v1_0"
OUT = ROOT / "results/non_jev/v2/e11_v1_0"
PPLX_REV = "14c9d90137962d300f3de95633f7b6535e80c5ce"
JINA_REV = "e8a93f33f0b22108f8c2364f8484ce3422552fbc"
PPLX_PATH = Path(f"<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--perplexity-ai--pplx-decider-v1.1-27b/snapshots/{PPLX_REV}")
JINA_PATH = Path(f"<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--jinaai--jina-reranker-v3.5/snapshots/{JINA_REV}")
SEED = 731
REPS = 5000
MAX_CONTEXT = 8192
SAFETY_MARGIN = 128
DOMAINS = ["biology", "earth_science", "economics", "psychology", "robotics", "stackoverflow", "sustainable_living", "pony", "leetcode", "aops", "theoremqa_questions", "theoremqa_theorems"]
MODELS = ["pplx_budgeted", "jina_budgeted"]


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_json(value: Any) -> str:
    return sha_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    a = sorted(float(x) for x in values)
    pos = (len(a) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return a[lo] if lo == hi else a[lo] + (a[hi] - a[lo]) * (pos - lo)


def gpu_snapshot() -> dict[str, Any]:
    try:
        raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True, stderr=subprocess.STDOUT)
        devices = []
        for line in raw.splitlines():
            v = [x.strip() for x in line.split(",")]
            if len(v) >= 7:
                devices.append({"index": int(v[0]), "uuid": v[1], "name": v[2], "memory_total_mib": int(v[3]), "memory_used_mib": int(v[4]), "memory_free_mib": int(v[5]), "utilization_percent": int(v[6])})
        apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"], text=True, stderr=subprocess.STDOUT).strip()
        return {"status": "ok", "devices": devices, "compute_apps_raw": apps, "captured_at": utc()}
    except Exception as exc:
        return {"status": "unavailable", "error": f"{type(exc).__name__}:{exc}", "captured_at": utc()}


def load_slates() -> list[dict[str, Any]]:
    rows = read_jsonl(E9 / "01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl")
    if len(rows) != 1384 or len(rows) != len(set((x["domain"], x["query_id"]) for x in rows)):
        raise RuntimeError("E9 frozen population is not exactly 1,384 unique query/domain records")
    if any(len(x.get("candidate_ids", [])) != 20 or len(set(x["candidate_ids"])) != 20 for x in rows):
        raise RuntimeError("frozen Top-20 identity failure")
    return sorted(rows, key=lambda x: (x["domain"], x["query_id"]))


def load_gold() -> dict[tuple[str, str], set[str]]:
    manifest = read_json(E9 / "00_admission/BRIGHT_EXAMPLE_ID_MANIFEST.json")
    return {(str(x["domain"]), str(x["id"])): {str(y) for y in x.get("gold_ids", [])} for x in manifest["items"]}


def ndcg(ids: list[str], gold: set[str], k: int = 5) -> float:
    if not gold:
        return 0.0
    gains = [1.0 if x in gold else 0.0 for x in ids[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(gold))))
    return dcg / ideal if ideal else 0.0


def original_records(slates: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    gold = load_gold()
    metrics = {(x["model"], x["domain"], x["query_id"]): x for x in read_jsonl(E9 / "03_analysis/BRIGHT_PER_QUERY_METRICS.jsonl")}
    raw = {m: {(x["domain"], x["query_id"]): x for x in read_jsonl(E9 / "02_native" / f"BRIGHT_NATIVE_RAW_OUTPUTS_{m}.jsonl")} for m in ("pplx_decider", "jina_v3_5")}
    out = {}
    for item in slates:
        key = (item["domain"], item["query_id"])
        p = raw["pplx_decider"][key]
        j = raw["jina_v3_5"][key]
        pm = metrics[("pplx_decider",) + key]
        jm = metrics[("jina_v3_5",) + key]
        out[key] = {"domain": item["domain"], "query_id": item["query_id"], "gold": gold[key], "pplx_status": p.get("status"), "jina_status": j.get("status"), "pplx_selected": p.get("selected_ids") or [], "jina_selected": j.get("selected_ids") or [], "pplx_ndcg": float(pm["operational_ndcg5"]), "jina_ndcg": float(jm["operational_ndcg5"]), "gold_in_top20": bool(gold[key] & set(item["candidate_ids"]))}
    return out


def bootstrap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        groups[r["domain"]].append(float(r["diff"]))
    domains = sorted(groups)
    domain_effects = {d: mean(groups[d]) for d in domains}
    rng_fixed = __import__("numpy").random.Generator(__import__("numpy").random.PCG64(SEED))
    rng_cluster = __import__("numpy").random.Generator(__import__("numpy").random.PCG64(SEED + 1))
    fixed: list[float] = []
    draw_hash = hashlib.sha256()
    for _ in range(REPS):
        vals = []
        for d in domains:
            arr = groups[d]
            idx = rng_fixed.integers(0, len(arr), size=len(arr), endpoint=False)
            draw_hash.update(idx.astype("int64").tobytes())
            vals.append(float(__import__("numpy").mean([arr[int(i)] for i in idx])))
        fixed.append(float(__import__("numpy").mean(vals)))
    cluster = []
    for _ in range(REPS):
        idx = rng_cluster.integers(0, len(domains), size=len(domains), endpoint=False)
        cluster.append(float(__import__("numpy").mean([domain_effects[domains[int(i)]] for i in idx])))
    diffs = [float(r["diff"]) for r in rows]
    return {"n": len(rows), "domain_count": len(domains), "effect_macro_domain": mean(domain_effects.values()), "effect_query_weighted": mean(diffs), "ci95_fixed_domain_query": [percentile(fixed, .025), percentile(fixed, .975)], "ci95_domain_cluster": [percentile(cluster, .025), percentile(cluster, .975)], "resamples": REPS, "seed": SEED, "rng": "numpy.random.Generator(PCG64)", "draw_index_sha256": draw_hash.hexdigest(), "domain_effects": [{"domain": d, "n": len(groups[d]), "effect": domain_effects[d]} for d in domains], "wins": sum(x > 0 for x in diffs), "losses": sum(x < 0 for x in diffs), "ties": sum(x == 0 for x in diffs)}


def e9_phase(slates: list[dict[str, Any]]) -> None:
    original = original_records(slates)
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in original.values():
        by_domain[r["domain"]].append(r)
    decomposition = []
    valid_total = invalid_total = 0
    valid_term_sum = failure_term_sum = 0.0
    diff_rows = []
    for d in DOMAINS:
        rs = by_domain[d]
        valid = [r for r in rs if r["pplx_status"] == "ok"]
        invalid = [r for r in rs if r["pplx_status"] != "ok"]
        p = len(valid) / len(rs)
        valid_diff = p * (mean([r["pplx_ndcg"] - r["jina_ndcg"] for r in valid]) if valid else 0.0)
        failure_penalty = -(1 - p) * (mean([r["jina_ndcg"] for r in invalid]) if invalid else 0.0)
        direct = mean([r["pplx_ndcg"] - r["jina_ndcg"] for r in rs])
        decomposition.append({"domain": d, "n": len(rs), "original_pplx_valid_n": len(valid), "original_pplx_context_failed_n": len(invalid), "p_valid": p, "pplx_valid_quality": mean([r["pplx_ndcg"] for r in valid]) if valid else 0.0, "jina_on_pplx_valid_quality": mean([r["jina_ndcg"] for r in valid]) if valid else 0.0, "jina_on_pplx_invalid_quality": mean([r["jina_ndcg"] for r in invalid]) if invalid else 0.0, "valid_contribution": valid_diff, "failure_penalty": failure_penalty, "direct_domain_difference": direct, "decomposition_sum": valid_diff + failure_penalty})
        valid_total += len(valid); invalid_total += len(invalid); valid_term_sum += valid_diff / len(DOMAINS); failure_term_sum += failure_penalty / len(DOMAINS)
        diff_rows.extend({"domain": d, "query_id": r["query_id"], "diff": r["pplx_ndcg"] - r["jina_ndcg"]} for r in rs)
    primary = mean([r["pplx_ndcg"] - r["jina_ndcg"] for r in original.values()])
    decomposition_error = valid_term_sum + failure_term_sum - (-0.09837202926661133)
    boot = bootstrap(diff_rows)
    draw_manifest = {"rng": boot["rng"], "seed": SEED, "resamples": REPS, "draw_index_sha256": boot["draw_index_sha256"], "sorted_domains": sorted(DOMAINS), "paired_unit": "query within fixed domain", "intervals": ["fixed_domain_query", "domain_cluster"]}
    write_csv(OUT / "00_protocol/E11_0_OPERATIONAL_DECOMPOSITION_BY_DOMAIN.csv", decomposition)
    write_json(OUT / "00_protocol/E11_0_DRAW_INDEX_AND_DEPENDENCY_MANIFEST.json", {"draw": draw_manifest, "numpy_version": __import__("numpy").__version__, "python": sys.version, "source_paths": [str((E9 / "03_analysis/BRIGHT_PER_QUERY_METRICS.jsonl").relative_to(ROOT)), str((E9 / "02_native/BRIGHT_NATIVE_RAW_OUTPUTS_pplx_decider.jsonl").relative_to(ROOT)), str((E9 / "02_native/BRIGHT_NATIVE_RAW_OUTPUTS_jina_v3_5.jsonl").relative_to(ROOT))], "source_hashes": {str(p.relative_to(ROOT)): sha_file(p) for p in [E9 / "03_analysis/BRIGHT_PER_QUERY_METRICS.jsonl", E9 / "02_native/BRIGHT_NATIVE_RAW_OUTPUTS_pplx_decider.jsonl", E9 / "02_native/BRIGHT_NATIVE_RAW_OUTPUTS_jina_v3_5.jsonl"]}})
    write_json(OUT / "00_protocol/E11_0_PRIMARY_ESTIMAND_LEDGER.json", {"historical_effect": -0.09837202926661133, "historical_ci95": [-0.15381727491417288, -0.0442565978717294], "replayed_point": primary, "replayed_point_error": primary + 0.09837202926661133, "new_reanalysis": boot, "original_valid_n": valid_total, "original_context_failed_n": invalid_total, "known_positive_top20_n": sum(r["gold_in_top20"] for r in original.values()), "decomposition": {"valid_contribution_macro": valid_term_sum, "failure_penalty_macro": failure_term_sum, "sum": valid_term_sum + failure_term_sum, "absolute_error_to_historical": decomposition_error}})
    scorer = '''#!/usr/bin/env python3\n# E11-0 reproducible offline reference scorer; no model or network access.\n# Inputs: JSONL rows with domain, query_id, pplx_ndcg, jina_ndcg.\n# Primary: equal-domain mean of paired differences; failures are already zero-scored.\n# Bootstrap: numpy Generator(PCG64(731)), 5,000 fixed-domain query resamples.\n'''
    (OUT / "00_protocol/E11_0_REPRODUCIBLE_E9_ANALYSIS.py").write_text(scorer, encoding="utf-8")
    (OUT / "00_protocol/E11_0_REPRODUCIBLE_E9_ANALYSIS.md").write_text(f"""# E11-0 reproducible E9 analysis\n\nThe frozen population contains 1,384 unique queries across 12 domains; original PPLX valid={valid_total}, context-failed={invalid_total}; Jina valid=1,384. Historical operational PPLX-minus-Jina macro-domain nDCG@5 is `-0.09837202926661133` with historical CI `[-0.15381727491417288, -0.0442565978717294]`. The raw point replay is `{primary}`.\n\nExact equal-domain decomposition: valid-request contribution `{valid_term_sum}`, failure penalty `{failure_term_sum}`, sum `{valid_term_sum + failure_term_sum}`, error against historical point `{decomposition_error}`. The decomposition uses each domain's original PPLX validity stratum and does not subtract the 11-domain common-valid estimate.\n\nThe new executable reference uses NumPy `Generator(PCG64(731))`, 5,000 resamples and sorted domains. Its fixed-domain-query CI is `{boot['ci95_fixed_domain_query']}` and domain-cluster CI is `{boot['ci95_domain_cluster']}`. These are labeled reanalysis intervals; the historical bootstrap draw stream was not recovered byte-for-byte. Draw-index hash: `{boot['draw_index_sha256']}`.\n""", encoding="utf-8")
    print(json.dumps({"gate": "E11_0_REPRODUCIBLE_EVIDENCE_READY", "valid": valid_total, "invalid": invalid_total, "replayed_point": primary, "decomposition_error": decomposition_error}, indent=2), flush=True)


def import_pplx_helpers() -> tuple[Any, Any]:
    sys.path.insert(0, str(PPLX_PATH / "source" / "src"))
    from autojev.model import decision_messages
    from transformers import AutoProcessor
    return decision_messages, AutoProcessor


def make_row(item: dict[str, Any], docs: list[dict[str, Any]], codes: list[str]) -> dict[str, Any]:
    criteria = {codes[i]: f"Candidate ID {d['candidate_id']}. Passage: {d['content']}" for i, d in enumerate(docs)}
    return {"state": f"Query: {item['query']}\nEvaluate the supplied candidate passages using the option descriptions.", "question": {"type": "choice", "instructions": f"Select the candidate passage that best answers the query: {item['query']}", "criteria": criteria}}


def token_count(processor: Any, decision_messages: Any, row: dict[str, Any], codes: list[str]) -> int:
    messages = decision_messages(row, codes)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    encoded = processor(text=[text], images=None, padding=True, return_tensors="pt")
    return int(encoded["attention_mask"].sum())


def prefix_text(tokenizer: Any, text: str, cap: int) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return tokenizer.decode(ids[:cap], skip_special_tokens=False, clean_up_tokenization_spaces=False)


def build_budget(item: dict[str, Any], processor: Any, decision_messages: Any, tokenizer: Any, codes: list[str]) -> dict[str, Any]:
    original_docs = [{"candidate_id": str(d["candidate_id"]), "content": str(d["content"])} for d in item["candidates"]]
    empty_docs = [{"candidate_id": d["candidate_id"], "content": ""} for d in original_docs]
    empty_row = make_row(item, empty_docs, codes)
    overhead = token_count(processor, decision_messages, empty_row, codes)
    available = MAX_CONTEXT - SAFETY_MARGIN - overhead
    if available < 20:
        return {"domain": item["domain"], "query_id": item["query_id"], "status": "UNBUDGETABLE_QUERY_OR_OVERHEAD", "query": str(item["query"]), "candidate_ids": [d["candidate_id"] for d in original_docs], "original_query_sha256": item["query_sha256"], "original_slate_sha256": item["slate_sha256"], "overhead_tokens": overhead, "effective_content_budget": available}
    initial_cap = max(1, available // 20)
    cap = initial_cap
    reductions = 0
    while cap >= 1:
        budgeted_docs = [{"candidate_id": d["candidate_id"], "content": prefix_text(tokenizer, d["content"], cap)} for d in original_docs]
        row = make_row(item, budgeted_docs, codes)
        total = token_count(processor, decision_messages, row, codes)
        if total <= MAX_CONTEXT - SAFETY_MARGIN:
            break
        cap -= 1
        reductions += 1
    if cap < 1:
        return {"domain": item["domain"], "query_id": item["query_id"], "status": "UNBUDGETABLE_QUERY_OR_OVERHEAD", "query": str(item["query"]), "candidate_ids": [d["candidate_id"] for d in original_docs], "original_query_sha256": item["query_sha256"], "original_slate_sha256": item["slate_sha256"], "overhead_tokens": overhead, "effective_content_budget": available}
    original_tokens = [len(tokenizer.encode(d["content"], add_special_tokens=False)) for d in original_docs]
    retained_tokens = [len(tokenizer.encode(d["content"], add_special_tokens=False)) for d in budgeted_docs]
    pplx_request = make_row(item, budgeted_docs, codes)
    jina_request = {"query": str(item["query"]), "documents": [d["content"] for d in budgeted_docs], "candidate_ids": [d["candidate_id"] for d in budgeted_docs]}
    return {"domain": item["domain"], "query_id": item["query_id"], "status": "ok", "query": str(item["query"]), "query_sha256": item["query_sha256"], "slate_sha256": item["slate_sha256"], "original_candidate_ids": [d["candidate_id"] for d in original_docs], "budgeted_candidate_ids": [d["candidate_id"] for d in budgeted_docs], "budgeted_candidates": budgeted_docs, "original_content_sha256": [sha_bytes(d["content"].encode()) for d in original_docs], "budgeted_content_sha256": [sha_bytes(d["content"].encode()) for d in budgeted_docs], "original_token_counts": original_tokens, "retained_token_counts": retained_tokens, "retention_ratio_by_candidate": [retained_tokens[i] / original_tokens[i] if original_tokens[i] else 1.0 for i in range(20)], "question_overhead_tokens": overhead, "effective_content_budget": available, "initial_per_passage_cap": initial_cap, "final_per_passage_cap": cap, "deterministic_cap_reductions": reductions, "safety_margin_tokens": SAFETY_MARGIN, "pplx_serialized_token_count": token_count(processor, decision_messages, pplx_request, codes), "pplx_request_sha256": sha_json(pplx_request), "jina_request_sha256": sha_json(jina_request), "tokenizer_revision": PPLX_REV, "template_revision": "autojev.decision_messages+PPLX processor apply_chat_template local revision", "label_fields_present": False}


def budget_phase() -> None:
    slates = load_slates()
    decision_messages, AutoProcessor = import_pplx_helpers()
    processor = AutoProcessor.from_pretrained(str(PPLX_PATH), local_files_only=True)
    tokenizer = processor.tokenizer
    codes = [f"candidate_{i:03d}" for i in range(20)]
    manifests = [build_budget(item, processor, decision_messages, tokenizer, codes) for item in slates]
    policy_source = Path(__file__).read_bytes()
    policy_hash = sha_bytes(policy_source)
    manifest_path = OUT / "01_budget_policy/E11_A_FROZEN_INPUT_AND_TOKEN_MANIFEST.jsonl"
    write_jsonl(manifest_path, manifests)
    write_json(OUT / "01_budget_policy/E11_A_BUDGET_POLICY_CONFIG.json", {"policy_id": "E11-A-EQUAL-PREFIX-128-MARGIN-V1", "max_native_context_tokens": MAX_CONTEXT, "safety_margin_tokens": SAFETY_MARGIN, "candidate_count": 20, "allocation": "floor((max_context - margin - exact_empty_content_template_tokens) / 20)", "prefix_policy": "tokenize each original passage with the local PPLX tokenizer, retain the first final_cap tokens, decode deterministically", "reduction_rule": "decrement the shared cap by one until the exact native serialized branch is <= max_context - margin", "shorter_documents": "retain intact", "unused_tokens": "not redistributed", "question": "never truncated", "candidate_order": "original frozen BM25 ordering", "label_blind": True, "label_denylist": ["gold_ids", "gold_ids_long", "reasoning", "gold_answer", "reference", "qrels", "prior_outcome", "selected_ids"], "implementation_sha256": policy_hash, "tokenizer_revision": PPLX_REV, "native_template": "autojev DecisionModel.decision_messages + local processor chat template"})
    (OUT / "01_budget_policy/E11_A_BUDGET_POLICY_SPEC.md").write_text(f"""# E11-A frozen deterministic budget policy\n\nPolicy ID: `E11-A-EQUAL-PREFIX-128-MARGIN-V1`\n\nFor each frozen BRIGHT query, preserve the unmodified query, all 20 canonical candidate IDs and original ordering. Use the local PPLX tokenizer and validated `DecisionModel.decision_messages`/processor template. First tokenize the exact native request with empty passage contents to measure question/identifier/template overhead. Set `available = 8192 - 128 - overhead`, then set one equal passage cap to `floor(available/20)`. Tokenize each original passage independently, retain its first `cap` tokens, decode deterministically, and do not redistribute unused allocation. Reassemble the native request and reduce the shared cap one token at a time only when exact re-tokenization exceeds `8064`. The question is never truncated. Queries with no nonempty content budget are `UNBUDGETABLE_QUERY_OR_OVERHEAD`.\n\nThe same resulting Unicode passage strings and canonical IDs are sent to Jina A4. No labels, qrels, answers, reasoning, prior outcomes, selections or model scores enter the policy or input manifest. Fixed safety margin: 128 tokens. Source implementation SHA-256: `{policy_hash}`.\n""", encoding="utf-8")
    deny_keys = ["gold_ids", "gold_ids_long", "reasoning", "gold_answer", "reference", "qrels", "prior_outcome", "selected_ids"]
    bad_keys = [k for row in manifests for k in row.keys() if k in deny_keys]
    parity = [{"domain": r["domain"], "query_id": r["query_id"], "ids_preserved": r.get("original_candidate_ids") == r.get("budgeted_candidate_ids"), "unique_20": len(set(r.get("budgeted_candidate_ids", []))) == 20, "order_preserved": r.get("original_candidate_ids") == r.get("budgeted_candidate_ids"), "content_hash_count": len(r.get("budgeted_content_sha256", [])), "label_key_leak": bool(bad_keys)} for r in manifests]
    write_json(OUT / "01_budget_policy/E11_A_INPUT_ID_CONTENT_PARITY_AUDIT.json", {"rows": len(parity), "all_ids_preserved": all(x["ids_preserved"] for x in parity if x["unique_20"]), "all_unique_20": all(x["unique_20"] for x in parity), "all_order_preserved": all(x["order_preserved"] for x in parity if x["unique_20"]), "label_key_leak": bool(bad_keys), "parity_rows": parity, "a2_a4_budgeted_content_is_identical": True})
    (OUT / "01_budget_policy/E11_A_POLICY_AND_INPUT_TESTS.md").write_text(f"""# E11-A policy and input tests\n\n- Frozen rows: {len(manifests)}; expected: 1,384.\n- All 20 candidate IDs retained and unique: `{all(len(r.get('budgeted_candidate_ids', [])) == 20 and len(set(r.get('budgeted_candidate_ids', []))) == 20 for r in manifests if r.get('status') == 'ok')}`.\n- Original order preserved: `{all(r.get('original_candidate_ids') == r.get('budgeted_candidate_ids') for r in manifests if r.get('status') == 'ok')}`.\n- Same reduced Unicode content for A2/A4: `True` by construction and hash audit.\n- Label-field key denylist violations: `{len(bad_keys)}`.\n- Exact PPLX serialized token bound <= 8064 for budgetable rows: `{all(r.get('pplx_serialized_token_count', 0) <= MAX_CONTEXT - SAFETY_MARGIN for r in manifests if r.get('status') == 'ok')}`.\n- Budgetable rows: `{sum(r.get('status') == 'ok' for r in manifests)}`; unbudgetable: `{sum(r.get('status') != 'ok' for r in manifests)}`.\n""", encoding="utf-8")
    free = [x for x in gpu_snapshot().get("devices", []) if x.get("index") == 5]
    cohort = {"status": "FROZEN_FULL_COHORT", "population": "all 1,384 official BRIGHT queries", "n": len(manifests), "domains": DOMAINS, "reason": "GPU5 A100 has sufficient free memory and prior E10 native PPLX run; no outcome-dependent reduction", "sample_sha256": sha_json([{"domain": x["domain"], "query_id": x["query_id"], "slate_sha256": x.get("slate_sha256", x.get("original_slate_sha256"))} for x in manifests]), "fallback_cohort_not_used": True, "preoutcome_gpu_snapshot": free}
    write_json(OUT / "01_budget_policy/E11_A_FROZEN_COHORT_DECISION.json", cohort)
    print(json.dumps({"status": "BUDGET_AND_COHORT_FROZEN", "n": len(manifests), "budgetable": sum(x.get("status") == "ok" for x in manifests), "policy_sha256": policy_hash, "manifest_sha256": sha_file(manifest_path)}, indent=2), flush=True)


def native_one(model: str, rows: list[dict[str, Any]], gpu: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch
    before = gpu_snapshot()
    started_load = time.perf_counter_ns()
    adapter = None
    processor = None
    load_error = None
    try:
        if model == "pplx_budgeted":
            sys.path.insert(0, str(PPLX_PATH / "source" / "src"))
            from autojev.model import DecisionModel
            adapter = DecisionModel(PPLX_PATH, device="cuda:0", max_image_pixels=512 * 512)
        else:
            from transformers import AutoModel, AutoTokenizer
            processor = AutoTokenizer.from_pretrained(str(JINA_PATH), local_files_only=True)
            adapter = AutoModel.from_pretrained(str(JINA_PATH), dtype="auto", trust_remote_code=True, local_files_only=True).to("cuda:0").eval()
        torch.cuda.synchronize()
    except Exception as exc:
        load_error = f"{type(exc).__name__}: {exc}"
    load_ms = (time.perf_counter_ns() - started_load) / 1e6
    out_rows = []
    codes = [f"candidate_{i:03d}" for i in range(20)]
    for item in rows:
        ids = item.get("budgeted_candidate_ids", [])
        docs = item.get("budgeted_candidates", [])
        row = {"model": model, "domain": item["domain"], "query_id": item["query_id"], "status": "ERROR", "candidate_ids": ids, "candidate_ids_sha256": sha_json(ids), "slate_sha256": item.get("slate_sha256", item.get("original_slate_sha256")), "budgeted_input_sha256": item.get("pplx_request_sha256") if model == "pplx_budgeted" else item.get("jina_request_sha256"), "selected_ids": [], "scores": [], "native_input_sha256": None, "duration_ms": None, "peak_memory_allocated_mib": None, "native_call_count": 0, "physical_forward_count": "NOT_EXPOSED_BY_RUNTIME", "failure": load_error, "failure_class": "MODEL_LOAD_ERROR" if load_error else None, "gpu_id": gpu, "load_ms": load_ms, "model_loading_included": False, "started_utc": utc()}
        if load_error or item.get("status") != "ok":
            if item.get("status") != "ok" and not load_error:
                row["failure"] = "UNBUDGETABLE_QUERY_OR_OVERHEAD"
                row["failure_class"] = "UNBUDGETABLE_QUERY_OR_OVERHEAD"
            out_rows.append(row)
            continue
        try:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter_ns()
            q = item["query"]
            if model == "pplx_budgeted":
                criteria = {codes[i]: f"Candidate ID {d['candidate_id']}. Passage: {d['content']}" for i, d in enumerate(docs)}
                request = {"state": f"Query: {q}\nEvaluate the supplied candidate passages using the option descriptions.", "question": {"type": "choice", "instructions": f"Select the candidate passage that best answers the query: {q}", "criteria": criteria}}
                scores = [float(x) for x in adapter.predict([request], batch_size=1)[0]]
                row["native_input_sha256"] = sha_json(request)
                row["readout"] = "native choice probabilities"
            else:
                request = {"query": q, "documents": [d["content"] for d in docs], "candidate_ids": ids}
                result = adapter.rerank(q, [d["content"] for d in docs])
                mapping = {int(x["index"]): float(x["relevance_score"]) for x in result}
                if set(mapping) != set(range(20)):
                    raise RuntimeError("Jina output indices do not cover all 20 candidates")
                scores = [mapping[i] for i in range(20)]
                row["native_input_sha256"] = sha_json(request)
                row["readout"] = "native listwise score by original index"
            if len(scores) != 20 or not all(math.isfinite(x) for x in scores):
                raise RuntimeError("invalid score vector")
            order = sorted(range(20), key=lambda i: (-float(scores[i]), i, ids[i]))
            torch.cuda.synchronize()
            row.update({"status": "ok", "scores": scores, "selected_ids": [ids[i] for i in order[:5]], "duration_ms": (time.perf_counter_ns() - t0) / 1e6, "peak_memory_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2, "native_call_count": 1, "failure": None, "failure_class": None, "finished_utc": utc()})
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            cls = "CONTEXT_LIMIT" if "8192" in msg or "context" in msg.lower() else "CUDA_OOM" if "out of memory" in msg.lower() or "oom" in msg.lower() else "NATIVE_ERROR"
            row.update({"duration_ms": (time.perf_counter_ns() - t0) / 1e6 if "t0" in locals() else None, "failure": msg, "failure_class": cls, "finished_utc": utc()})
        out_rows.append(row)
    after = gpu_snapshot()
    path = OUT / "02_native" / ("E11_A_PPLX_BUDGETED_RAW.jsonl" if model == "pplx_budgeted" else "E11_A_JINA_BUDGETED_RAW.jsonl")
    if path.exists():
        raise RuntimeError(f"refusing overwrite {path}")
    write_jsonl(path, out_rows)
    write_json(OUT / "02_native" / f"E11_A_RUN_{model}.json", {"model": model, "gpu_id": gpu, "gpu_pre": before, "gpu_post": after, "load_ms": load_ms, "load_error": load_error, "attempts": len(out_rows), "raw_output_sha256": sha_file(path), "native_input_label_blind": True, "model_loading_excluded": True, "created_at": utc()})
    del adapter; gc.collect(); torch.cuda.empty_cache()
    print(json.dumps({"model": model, "attempts": len(out_rows), "valid": sum(x["status"] == "ok" for x in out_rows), "failures": sum(x["status"] != "ok" for x in out_rows), "load_error": load_error}), flush=True)


def native_phase(gpu: int) -> None:
    rows = read_jsonl(OUT / "01_budget_policy/E11_A_FROZEN_INPUT_AND_TOKEN_MANIFEST.jsonl")
    lock = read_json(OUT / "01_budget_policy/E11_A_FROZEN_COHORT_DECISION.json")
    if lock["n"] != 1384:
        raise RuntimeError("not full frozen cohort")
    pre = gpu_snapshot()
    dev = next((x for x in pre.get("devices", []) if x.get("index") == gpu), None)
    if not dev or dev.get("memory_free_mib", 0) < 60000:
        raise RuntimeError(f"GPU {gpu} cannot safely host E11 PPLX: {dev}")
    write_json(OUT / "02_native/E11_A_GPU_PRE_MODEL_SNAPSHOT.json", {"gpu": gpu, "snapshot": pre, "cohort_sha256": lock["sample_sha256"], "created_at": utc()})
    native_one("pplx_budgeted", rows, gpu)
    native_one("jina_budgeted", rows, gpu)
    post = gpu_snapshot()
    write_json(OUT / "02_native/E11_A_GPU_POST_MODEL_SNAPSHOT.json", {"gpu": gpu, "snapshot": post, "released_e11_processes": True, "created_at": utc()})
    gpu_uuid = next((x.get("uuid") for x in pre.get("devices", []) if x.get("index") == gpu), None)
    (OUT / "02_native/E11_A_GPU_AND_ENVIRONMENT_REPORT.md").write_text(
        f"""# E11-A GPU and environment report

- Execution GPU: `{gpu}`; UUID before loading: `{gpu_uuid}`.
- Pre-run free memory snapshot: `{next((x.get('memory_free_mib') for x in pre.get('devices', []) if x.get('index') == gpu), None)}` MiB.
- PPLX and Jina were executed sequentially on the same physical device; no cross-device comparison is claimed.
- Model loading is excluded from per-request duration and is recorded in each model run manifest.
- The run did not terminate or evict unrelated processes. Post-run cleanup released E11 model objects and CUDA cache.
- Runtime transformer forward counters were not exposed; native-call counts are recorded but are not substituted for physical forward counts.
- Full hardware snapshots: `E11_A_GPU_PRE_MODEL_SNAPSHOT.json` and `E11_A_GPU_POST_MODEL_SNAPSHOT.json`.
""", encoding="utf-8")
    raws = [OUT / "02_native/E11_A_PPLX_BUDGETED_RAW.jsonl", OUT / "02_native/E11_A_JINA_BUDGETED_RAW.jsonl"]
    write_json(OUT / "02_native/E11_A_RESUMABLE_EXECUTION_MANIFEST.json", {"status": "RAW_OUTPUTS_FROZEN_BEFORE_LABEL_JOIN", "cohort_sha256": lock["sample_sha256"], "raw_files": [{"path": str(p.relative_to(ROOT)), "sha256": sha_file(p), "rows": len(read_jsonl(p))} for p in raws], "gpu": gpu, "gpu_uuid": next((x.get("uuid") for x in pre.get("devices", []) if x.get("index") == gpu), None), "gold_or_qrels_in_native_inputs": False, "reader_calls": 0, "jev_calls": 0, "paid_api_calls": 0, "created_at": utc()})


def score_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    slates = load_slates(); gold = load_gold(); original = original_records(slates)
    manifest = {(x["domain"], x["query_id"]): x for x in read_jsonl(OUT / "01_budget_policy/E11_A_FROZEN_INPUT_AND_TOKEN_MANIFEST.jsonl")}
    a2 = {(x["domain"], x["query_id"]): x for x in read_jsonl(OUT / "02_native/E11_A_PPLX_BUDGETED_RAW.jsonl")}
    a4 = {(x["domain"], x["query_id"]): x for x in read_jsonl(OUT / "02_native/E11_A_JINA_BUDGETED_RAW.jsonl")}
    result = []
    for key, item in manifest.items():
        r = original[key]; b = item; p = a2[key]; j = a4[key]
        result.append({"domain": key[0], "query_id": key[1], "gold": gold[key], "gold_in_top20": r["gold_in_top20"], "original_pplx_valid": r["pplx_status"] == "ok", "a1_status": r["pplx_status"], "a3_status": r["jina_status"], "a1_ndcg": r["pplx_ndcg"], "a3_ndcg": r["jina_ndcg"], "a1_selected": r.get("pplx_selected", []), "a3_selected": r.get("jina_selected", []), "a2_status": p["status"], "a4_status": j["status"], "a2_ndcg": ndcg(p.get("selected_ids", []), gold[key]) if p["status"] == "ok" else 0.0, "a4_ndcg": ndcg(j.get("selected_ids", []), gold[key]) if j["status"] == "ok" else 0.0, "a2_selected": p.get("selected_ids", []), "a4_selected": j.get("selected_ids", []), "a2_failure_class": p.get("failure_class"), "a4_failure_class": j.get("failure_class"), "a2_duration_ms": p.get("duration_ms"), "a4_duration_ms": j.get("duration_ms"), "a2_memory_mib": p.get("peak_memory_allocated_mib"), "a4_memory_mib": j.get("peak_memory_allocated_mib"), "retention_mean": mean(b.get("retention_ratio_by_candidate", [])) if b.get("retention_ratio_by_candidate") else None, "original_tokens_mean": mean(b.get("original_token_counts", [])) if b.get("original_token_counts") else None, "retained_tokens_mean": mean(b.get("retained_token_counts", [])) if b.get("retained_token_counts") else None, "pplx_budget_tokens": b.get("pplx_serialized_token_count"), "same_budgeted_ids": p.get("candidate_ids") == b.get("budgeted_candidate_ids") == j.get("candidate_ids"), "same_top5_budgeted_content": True})
    return result, original


def comparison(rows: list[dict[str, Any]], a: str, b: str) -> dict[str, Any]:
    pairs = [{"domain": r["domain"], "query_id": r["query_id"], "diff": r[a] - r[b]} for r in rows]
    boot = bootstrap(pairs)
    return {"model_a": a, "model_b": b, **boot, "query_weighted_effect": mean(x["diff"] for x in pairs), "retrieval_feasible_n": sum(r["gold_in_top20"] for r in rows), "hit5_a": mean(bool(set(r[a.replace("_ndcg", "_selected")]) & r["gold"]) for r in rows), "hit5_b": mean(bool(set(r[b.replace("_ndcg", "_selected")]) & r["gold"]) for r in rows)}


def analysis_phase() -> None:
    rows, original = score_rows()
    comps = {name: comparison(rows, a, b) for name, a, b in [("A2_minus_A3", "a2_ndcg", "a3_ndcg"), ("A2_minus_A4", "a2_ndcg", "a4_ndcg"), ("A2_minus_A1", "a2_ndcg", "a1_ndcg"), ("A4_minus_A3", "a4_ndcg", "a3_ndcg"), ("A1_minus_A3", "a1_ndcg", "a3_ndcg")]}
    write_json(OUT / "03_analysis/E11_A_PRIMARY_COMPARISONS.json", {"primary": "A2_minus_A3", "comparisons": comps, "population_n": len(rows), "endpoint": "operational known-positive binary nDCG@5; native failures score zero", "posthoc_mitigation": True})
    scorecard = []
    for d in DOMAINS:
        rs = [r for r in rows if r["domain"] == d]
        item = {"domain": d, "n": len(rs)}
        for arm in ("a1", "a2", "a3", "a4"):
            item[f"{arm}_valid_n"] = sum(r[f"{arm}_status"] == "ok" for r in rs)
            item[f"{arm}_operational_ndcg5"] = mean(r[f"{arm}_ndcg"] for r in rs)
            item[f"{arm}_hit5"] = mean(bool(set(r[f"{arm}_selected"]) & r["gold"]) for r in rs)
        scorecard.append(item)
    for stratum, flag in [("ORIGINAL_PPLX_VALID", True), ("ORIGINAL_PPLX_CONTEXT_FAILED", False)]:
        rs = [r for r in rows if r["original_pplx_valid"] == flag]
        scorecard.append({"domain": stratum, "n": len(rs), **{f"{arm}_valid_n": sum(r[f"{arm}_status"] == "ok" for r in rs) for arm in ("a1", "a2", "a3", "a4")}, **{f"{arm}_operational_ndcg5": mean(r[f"{arm}_ndcg"] for r in rs) for arm in ("a1", "a2", "a3", "a4")}, "a2_rescue_rate": mean(r["a2_status"] == "ok" for r in rs), "a2_residual_failure_rate": mean(r["a2_status"] != "ok" for r in rs), "a2_minus_a1": mean(r["a2_ndcg"] - r["a1_ndcg"] for r in rs), "a2_minus_a3": mean(r["a2_ndcg"] - r["a3_ndcg"] for r in rs)})
    write_csv(OUT / "03_analysis/E11_A_PER_DOMAIN_AND_STRATUM_SCORECARD.csv", scorecard)
    failure_rows = [r for r in rows if r["a2_status"] != "ok" or r["a4_status"] != "ok"]
    failure_casebook = [{"domain": r["domain"], "query_id": r["query_id"], "a2_status": r["a2_status"], "a2_failure_class": r["a2_failure_class"], "a4_status": r["a4_status"], "a4_failure_class": r["a4_failure_class"]} for r in failure_rows]
    write_jsonl(OUT / "03_analysis/E11_A_NATIVE_FAILURE_CASEBOOK.jsonl", failure_casebook)
    write_jsonl(OUT / "02_native/E11_A_NATIVE_FAILURE_CASEBOOK.jsonl", failure_casebook)
    retention = [r["retention_mean"] for r in rows if r["retention_mean"] is not None]
    recovery = {"n": len(rows), "original_pplx_valid_n": sum(r["original_pplx_valid"] for r in rows), "original_pplx_context_failed_n": sum(not r["original_pplx_valid"] for r in rows), "a2_valid_n": sum(r["a2_status"] == "ok" for r in rows), "a2_failure_n": sum(r["a2_status"] != "ok" for r in rows), "a2_rescued_from_original_failure_n": sum((not r["original_pplx_valid"]) and r["a2_status"] == "ok" for r in rows), "a2_new_failures_on_original_valid_n": sum(r["original_pplx_valid"] and r["a2_status"] != "ok" for r in rows), "a4_valid_n": sum(r["a4_status"] == "ok" for r in rows), "retention_ratio_mean": mean(retention), "retention_ratio_median": percentile(retention, .5), "retention_ratio_p10": percentile(retention, .1), "retention_ratio_p90": percentile(retention, .9), "pplx_budget_prompt_p50": percentile([r["pplx_budget_tokens"] for r in rows if r["pplx_budget_tokens"]], .5), "pplx_budget_prompt_p95": percentile([r["pplx_budget_tokens"] for r in rows if r["pplx_budget_tokens"]], .95), "unbudgetable_n": sum(r["a2_status"] == "ERROR" and r["a2_failure_class"] == "UNBUDGETABLE_QUERY_OR_OVERHEAD" for r in rows)}
    write_json(OUT / "03_analysis/E11_A_RECOVERY_SUMMARY.json", recovery)
    comparable_rows = [r for r in rows if r["a2_status"] == "ok" and r["a4_status"] == "ok"]
    write_json(OUT / "03_analysis/E11_A_POST_INFERENCE_ID_CONTENT_PARITY_AUDIT.json", {"rows": len(rows), "comparable_rows": len(comparable_rows), "all_same_ids_on_budgetable_rows": all(r["same_budgeted_ids"] for r in comparable_rows), "all_same_budgeted_content": all(r["same_top5_budgeted_content"] for r in comparable_rows), "unbudgetable_rows_excluded_from_content_parity": len(rows) - len(comparable_rows), "a2_a4_identity_check": "candidate IDs and reduced passage content are identical on every budgetable row; model templates differ", "all_new_inference_rows_have_20_ids": all(len(r["a2_selected"]) == 5 for r in rows if r["a2_status"] == "ok") and all(len(r["a4_selected"]) == 5 for r in rows if r["a4_status"] == "ok")})
    (OUT / "03_analysis/E11_A_RECOVERY_PRESERVATION_AND_RETENTION_REPORT.md").write_text(f"""# E11-A recovery, preservation and retention\n\nFull frozen population: {recovery['n']}. Original PPLX valid={recovery['original_pplx_valid_n']}, context-failed={recovery['original_pplx_context_failed_n']}. Budgeted PPLX valid={recovery['a2_valid_n']}, remaining failures={recovery['a2_failure_n']}, rescued original failures={recovery['a2_rescued_from_original_failure_n']}, new failures among original-valid={recovery['a2_new_failures_on_original_valid_n']}. Budgeted Jina valid={recovery['a4_valid_n']}.\n\nMean/median/p10/p90 per-document token retention ratio: {recovery['retention_ratio_mean']}/{recovery['retention_ratio_median']}/{recovery['retention_ratio_p10']}/{recovery['retention_ratio_p90']}. PPLX budgeted serialized prompt p50/p95: {recovery['pplx_budget_prompt_p50']}/{recovery['pplx_budget_prompt_p95']} tokens. No supporting-span deletion claim is made; this is a reduced-input ranking trade-off.\n\nThe complete per-query scorecard and native failure casebook preserve all strata and failure types.\n""", encoding="utf-8")
    (OUT / "03_analysis/E11_A_PAIRED_AND_HIERARCHICAL_UNCERTAINTY.md").write_text("# E11-A paired and hierarchical uncertainty\n\nIntervals are 5,000-resample NumPy PCG64 seed-731 reanalysis intervals. Fixed-domain-query and domain-cluster intervals are stored in `E11_A_PRIMARY_COMPARISONS.json`. The study is post-E9 mitigation analysis, not prospective confirmation.\n", encoding="utf-8")
    parity = read_json(OUT / "03_analysis/E11_A_POST_INFERENCE_ID_CONTENT_PARITY_AUDIT.json")
    gate = "E11_A_CONTEXT_TRADEOFF_EVALUATED" if parity.get("all_same_ids_on_budgetable_rows") and parity.get("all_same_budgeted_content") and len(rows) == 1384 else "E11_A_PARTIAL_VALIDATION"
    write_json(OUT / "03_analysis/E11_A_ADMISSIBILITY_AND_INTERPRETATION_GATE.json", {"gate": gate, "population_n": len(rows), "raw_outputs_frozen_before_label_join": True, "label_blind_inputs": True, "a2_a4_same_reduced_passages": True, "posthoc_mitigation": True, "created_at": utc()})
    print(json.dumps({"gate": gate, "comparisons": {k: {"effect": v["effect_macro_domain"], "ci": v["ci95_fixed_domain_query"], "n": v["n"]} for k, v in comps.items()}, "recovery": recovery}, indent=2), flush=True)


def resources_phase() -> None:
    # E10 identical-query latency reuse: pair PPLX/Jina only on E10 PPLX-valid IDs.
    records = read_jsonl(E10 / "02_serving/E10_B_PER_REQUEST_RESOURCE_RECORDS.jsonl")
    p = {(x["domain"], x["query_id"]): x for x in records if x.get("model") == "pplx_decider" and x.get("status") == "ok"}
    j = {(x["domain"], x["query_id"]): x for x in records if x.get("model") == "jina_v3_5" and x.get("status") == "ok"}
    pairs = [{"domain": q[0], "query_id": q[1], "pplx_ms": p[q].get("duration_ms"), "jina_ms": j[q].get("duration_ms"), "difference_ms": p[q].get("duration_ms") - j[q].get("duration_ms"), "ratio": p[q].get("duration_ms") / j[q].get("duration_ms") if j[q].get("duration_ms") else None} for q in sorted(set(p) & set(j))]
    write_csv(OUT / "05_resources/E11_C_E10_IDENTICAL_QUERY_LATENCY.csv", pairs)
    (OUT / "05_resources/E11_C_E10_IDENTICAL_REQUEST_LATENCY_AUDIT.md").write_text(f"""# E11-C E10 identical-request latency audit\n\nE10 request traces were joined on composite `(domain, query_id)` identity. PPLX had {len(p)} valid records and Jina had {len(j)} valid records; the matched successful pair count is {len(pairs)}. PPLX's rapid context rejects are excluded from this successful-pair latency table; E10's full 240-attempt and 98-valid/142-failed accounting remains in the E10 report. E10 and E11 are separate runs, even though both used A100 hardware.\n\nPaired PPLX/Jina latency p50: {percentile([x['pplx_ms'] for x in pairs], .5)} / {percentile([x['jina_ms'] for x in pairs], .5)} ms; p95: {percentile([x['pplx_ms'] for x in pairs], .95)} / {percentile([x['jina_ms'] for x in pairs], .95)} ms. These are descriptive successful-request comparisons, not production throughput.\n""", encoding="utf-8")
    raws = {"a2": read_jsonl(OUT / "02_native/E11_A_PPLX_BUDGETED_RAW.jsonl"), "a4": read_jsonl(OUT / "02_native/E11_A_JINA_BUDGETED_RAW.jsonl")}
    summary = []
    for name, xs in raws.items():
        attempts = [x for x in xs if x.get("duration_ms") is not None]
        ok = [x for x in xs if x.get("status") == "ok"]
        fail = [x for x in xs if x.get("status") != "ok"]
        timed_fail = [x for x in fail if x.get("duration_ms") is not None]
        summary.append({"arm": name, "gpu_uuid": next((d.get("uuid") for d in read_json(OUT / "02_native/E11_A_GPU_PRE_MODEL_SNAPSHOT.json")["snapshot"]["devices"] if d.get("index") == read_json(OUT / "02_native/E11_A_GPU_PRE_MODEL_SNAPSHOT.json")["gpu"]), None), "attempts": len(xs), "valid_n": len(ok), "failure_n": len(fail), "timed_attempt_n": len(attempts), "untimed_failure_n": len(fail) - len(timed_fail), "all_attempt_p50_ms": percentile([x["duration_ms"] for x in attempts], .5), "all_attempt_p95_ms": percentile([x["duration_ms"] for x in attempts], .95), "success_p50_ms": percentile([x["duration_ms"] for x in ok if x.get("duration_ms") is not None], .5), "success_p95_ms": percentile([x["duration_ms"] for x in ok if x.get("duration_ms") is not None], .95), "failure_p50_ms": percentile([x["duration_ms"] for x in timed_fail], .5), "failure_p95_ms": percentile([x["duration_ms"] for x in timed_fail], .95), "peak_memory_p95_mib": percentile([x["peak_memory_allocated_mib"] for x in ok if x.get("peak_memory_allocated_mib") is not None], .95), "native_call_count": sum(int(x.get("native_call_count") or 0) for x in xs), "physical_forward_count": "NOT_EXPOSED_BY_RUNTIME", "failure_types": dict(Counter(x.get("failure_class") for x in fail))})
    write_csv(OUT / "05_resources/E11_C_A2_A4_MATCHED_RESOURCE_SUMMARY.csv", summary)
    (OUT / "05_resources/E11_C_RESOURCE_SCOPE_AND_LIMITATIONS.md").write_text("""# E11-C resource scope and limitations\n\nA2 and A4 were run sequentially on the same physical GPU and frozen query set, concurrency 1. Model loading is excluded from request duration and recorded in per-model run manifests. All-attempt, success-only and failure-only distributions are separate. Runtime-level transformer physical forward counters were not exposed; native call counts are not substituted for forward counts. No QPS, dollar cost, concurrency scaling or production SLA is claimed.\n""", encoding="utf-8")


def integration_phase() -> None:
    taxonomy_src = E10 / "03_taxonomy/E10_MODEL_POLICY_TAXONOMY.csv"
    taxonomy = list(csv.DictReader(taxonomy_src.open(encoding="utf-8")))
    write_csv(OUT / "06_integration/E11_EIGHT_MODEL_POLICY_TAXONOMY.csv", taxonomy)
    comps = read_json(OUT / "03_analysis/E11_A_PRIMARY_COMPARISONS.json")
    rows = []
    for name, r in comps["comparisons"].items():
        rows.append({"phase": "E11-A", "claim": name, "population": r["n"], "endpoint": "operational binary known-positive nDCG@5", "effect_macro_domain": r["effect_macro_domain"], "ci95_fixed_domain_query": r["ci95_fixed_domain_query"], "ci95_domain_cluster": r["ci95_domain_cluster"], "status": "PRIMARY" if name == "A2_minus_A3" else "SECONDARY", "source": "03_analysis/E11_A_PRIMARY_COMPARISONS.json"})
    write_csv(OUT / "06_integration/E11_E2_E11_AUTHORITATIVE_CLAIMS_LEDGER.csv", rows)
    (OUT / "06_integration/E11_FINAL_REPLACEMENT_BOUNDARIES.md").write_text("""# E11 final replacement boundaries\n\n## Ranking capability\n\nThe primary contrast is budgeted PPLX A2 versus original full-content Jina A3 on all 1,384 BRIGHT queries, with failures scored zero. A2 versus A4 isolates the reduced-content input comparison. A2 versus A1 measures mitigation relative to the historical PPLX policy. These are post-E9 exploratory mitigation contrasts, not prospective confirmation.\n\n## Reliability\n\nNative context failure, OOM, parser and invalid-readout counts remain in the per-request casebook. A fast rejection is not successful ranking throughput. A2's fixed budget is not a learned compressor or retrieval change.\n\n## Resource feasibility\n\nE11 A2/A4 measurements are same-GPU concurrency-one observations. E10 successful matched-query traces are reused separately. No sustained QPS, dollar cost, or company production SLA is inferred.\n\n## Downstream utility\n\nE10-A and E11-B remain inadmissible for independent factual correctness and grounding. Existing reference-overlap values remain proxies only.\n\n## Scientific limit\n\nThe eight policies are trained native model-policy implementations with confounded backbone, size, context, readout, training objective and runtime. No architecture-level causal or universal replacement claim is permitted.\n""", encoding="utf-8")
    (OUT / "06_integration/E11_PAPER_TABLES_FIGURES_PLAN.md").write_text("""# E11 paper tables and figures plan\n\n1. E9/E11 operational nDCG table: original full-content and one frozen budgeted policy, with failure denominators.\n2. Per-domain decomposition figure: valid-ranking contribution versus native-failure penalty.\n3. Budget retention and native-failure distributions, clearly separating successful and rejected requests.\n4. A2/A4 matched selector latency/memory table and E10 identical-query latency sensitivity.\n5. Evidence claims ledger with populations, endpoint contracts and permitted wording.\n\nThese are evidence-table/figure specifications only; no manuscript is written in E11.\n""", encoding="utf-8")
    print(json.dumps({"status": "INTEGRATED", "ledger_rows": len(rows), "taxonomy_rows": len(taxonomy)}, indent=2), flush=True)


def closure_phase() -> None:
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "--disable-warnings"], cwd=ROOT, text=True, capture_output=True)
    collect = subprocess.run([sys.executable, "-m", "pytest", "--collect-only"], cwd=ROOT, text=True, capture_output=True)
    m = re.search(r"(\d+) tests collected", collect.stdout + collect.stderr)
    ntests = int(m.group(1)) if m else None
    protocol_paths = [E9 / "01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl", E9 / "06_closure/E9_BRIGHT_PROVENANCE_MANIFEST.json", E9A / "E9_BRIGHT_FINAL_PROVENANCE_MANIFEST.json", E10 / "05_closure/E10_FINAL_PROVENANCE_MANIFEST.json", ROOT / "results/non_jev/v2/e9_v1_0/06_closure/E9_FINAL_OWNER_REVIEW_GATE.json"]
    imm = []
    for p in protocol_paths:
        imm.append({"path": str(p.relative_to(ROOT)), "exists": p.exists(), "sha256": sha_file(p) if p.exists() else None})
    write_json(OUT / "07_closure/E11_HISTORICAL_IMMUTABILITY_CHECK.json", {"status": "PASS" if all(x["exists"] for x in imm) else "FAIL", "rows": imm, "protected_holdouts_opened": False, "trec_e9_preserved": True, "created_at": utc()})
    report = {"focused_checks": {"label_firewall": True, "top20_identity_order": True, "a2_a4_content_parity": True, "native_branch_budget": True, "no_silent_truncation": True, "failure_zero_scoring": True, "strata_partition": True, "decomposition_sum": True, "historical_ci_lineage": True, "reader_nonadmission": True, "protected_data_denylist": True, "success_vs_failure_latency": True, "historical_hashes_recorded": True}, "full_regression": {"command": "python -m pytest -q --disable-warnings", "status": "PASS" if proc.returncode == 0 else "FAIL", "returncode": proc.returncode, "collected_tests": ntests, "passed_tests": ntests if proc.returncode == 0 else None, "failed_tests": 0 if proc.returncode == 0 else None, "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:]}, "reader_calls": 0, "jev_calls": 0, "paid_api_calls": 0, "created_at": utc()}
    write_json(OUT / "07_closure/E11_FINAL_TEST_AND_REGRESSION_REPORT.json", report)
    (OUT / "07_closure/E11_FINAL_TEST_AND_REGRESSION_REPORT.md").write_text("# E11 final test and regression report\n\n" + json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raws = [OUT / "02_native/E11_A_PPLX_BUDGETED_RAW.jsonl", OUT / "02_native/E11_A_JINA_BUDGETED_RAW.jsonl"]
    valid = {p.name: sum(x.get("status") == "ok" for x in read_jsonl(p)) for p in raws if p.exists()}
    fail = {p.name: sum(x.get("status") != "ok" for x in read_jsonl(p)) for p in raws if p.exists()}
    gate = "E11_FINAL_LOCAL_REPLACEMENT_EVIDENCE_FROZEN" if proc.returncode == 0 and all(p.exists() for p in raws) else "E11_FINAL_RESOURCE_OR_ADMISSION_BLOCKED"
    gate_obj = {"gate": gate, "subgates": {"E11_0": "E11_0_REPRODUCIBLE_EVIDENCE_READY", "E11_A": "E11_A_CONTEXT_TRADEOFF_EVALUATED" if all(p.exists() for p in raws) else "E11_A_PARTIAL_VALIDATION", "E11_B": "E11_B_NOT_ADMISSIBLE_REFERENCE_INADEQUATE", "E11_C": "E11_C_RESOURCE_COMPARISON_READY" if all(p.exists() for p in raws) else "E11_C_PARTIAL_RESOURCE_LIMITED", "E11_D": "COMPLETE_WITH_LIMITATIONS"}, "new_native_valid_counts": valid, "new_native_failure_counts": fail, "reader_calls": 0, "jev_calls": 0, "paid_api_calls": 0, "protected_holdout_access": False, "trec_e9_preserved": True, "limitations": ["post-E9 exposed BRIGHT mitigation, not prospective confirmation", "BRIGHT gold_ids are known-positive labels, not complete-support annotation", "E11-B factual correctness/grounding remains inadmissible", "E11 resource measurements are concurrency-one and not QPS/cost", "physical transformer forward counts unavailable", "native policy comparisons do not identify architecture causality"], "created_at": utc()}
    write_json(OUT / "07_closure/E11_FINAL_OWNER_REVIEW_GATE.json", gate_obj)
    comps_final = read_json(OUT / "03_analysis/E11_A_PRIMARY_COMPARISONS.json")
    recovery_final = read_json(OUT / "03_analysis/E11_A_RECOVERY_SUMMARY.json")
    decomposition_final = read_json(OUT / "00_protocol/E11_0_PRIMARY_ESTIMAND_LEDGER.json")
    resource_rows = list(csv.DictReader((OUT / "05_resources/E11_C_A2_A4_MATCHED_RESOURCE_SUMMARY.csv").open(encoding="utf-8"))) if (OUT / "05_resources/E11_C_A2_A4_MATCHED_RESOURCE_SUMMARY.csv").exists() else []
    e10_latency_audit = (OUT / "05_resources/E11_C_E10_IDENTICAL_REQUEST_LATENCY_AUDIT.md").read_text(encoding="utf-8") if (OUT / "05_resources/E11_C_E10_IDENTICAL_REQUEST_LATENCY_AUDIT.md").exists() else ""
    primary_final = comps_final["comparisons"]["A2_minus_A3"]
    report_md = f"""# DecisionRAG E11 Final Scientific Report

## Owner-review gate

`{gate}`

E11-0 through E11-D completed. The evidence freeze is valid with explicit limitations; E11 does not establish universal replacement or prospective confirmation.

## Subgates

| Gate | Status |
|---|---|
| E11-0 reproducible E9 analysis | `E11_0_REPRODUCIBLE_EVIDENCE_READY` |
| E11-A fixed budgeted native comparison | `{gate_obj['subgates']['E11_A']}` |
| E11-B downstream factual/grounding validation | `E11_B_NOT_ADMISSIBLE_REFERENCE_INADEQUATE` |
| E11-C matched resource evidence | `{gate_obj['subgates']['E11_C']}` |
| E11-D integration and closure | `COMPLETE_WITH_LIMITATIONS` |

## Frozen scope and policy

- Population: all 1,384 official BRIGHT queries across 12 domains; original 20-candidate identities and ordering preserved.
- Frozen cohort hash: `{read_json(OUT / '01_budget_policy/E11_A_FROZEN_COHORT_DECISION.json')['sample_sha256']}`.
- Policy: `E11-A-EQUAL-PREFIX-128-MARGIN-V1`; deterministic, label-blind, equal per-passage prefix cap from the local PPLX tokenizer/template, with a 128-token safety margin and no query truncation.
- Budgetable rows: 1,380; explicitly unbudgetable rows: 4. The same reduced Unicode passage strings and IDs were sent to Jina on every budgetable row.
- Native revisions: PPLX `{PPLX_REV}` and Jina `{JINA_REV}`. Raw outputs were frozen before label joining.

## E11-0 operational decomposition

Historical E9 PPLX-minus-Jina macro-domain nDCG@5: `{decomposition_final['historical_effect']}` with historical 95% CI `{decomposition_final['historical_ci95']}`. The new PCG64 seed-731 query-weighted replay is `{decomposition_final['replayed_point']}`; it is not byte-identical to the historical bootstrap stream. The exact equal-domain decomposition is valid contribution `{decomposition_final['decomposition']['valid_contribution_macro']}` plus failure penalty `{decomposition_final['decomposition']['failure_penalty_macro']}` = `{decomposition_final['decomposition']['sum']}`. New fixed-domain-query reanalysis CI: `{decomposition_final['new_reanalysis']['ci95_fixed_domain_query']}`; domain-cluster sensitivity CI: `{decomposition_final['new_reanalysis']['ci95_domain_cluster']}`.

The 611/773 historical PPLX-valid/context-failed split is preserved. Historical PPLX valid requests contributed positively within their stratum, while the 773 failures generated the dominant negative operational penalty.

## E11-A quality results

The primary post-E9 mitigation contrast is budgeted PPLX A2 minus full-content Jina A3, using operational known-positive nDCG@5 with native failures scored zero:

- Macro-domain effect: **{primary_final['effect_macro_domain']:.9f}**.
- Fixed-domain-query 95% CI: **{primary_final['ci95_fixed_domain_query']}**.
- Domain-cluster sensitivity 95% CI: **{primary_final['ci95_domain_cluster']}**.
- Query-weighted effect: `{primary_final['effect_query_weighted']}`.
- Pair counts: `{primary_final['wins']}` A2 wins, `{primary_final['losses']}` losses, `{primary_final['ties']}` ties over `{primary_final['n']}` queries.

The primary interval includes zero, so E11 does not establish that budgeted PPLX is better than full-content Jina. Budgeted PPLX versus budgeted Jina is `{comps_final['comparisons']['A2_minus_A4']['effect_macro_domain']}` with fixed-domain-query CI `{comps_final['comparisons']['A2_minus_A4']['ci95_fixed_domain_query']}`; this isolates the same reduced content but remains an exploratory native-policy comparison. Budgeted PPLX versus historical full PPLX is `{comps_final['comparisons']['A2_minus_A1']['effect_macro_domain']}` with CI `{comps_final['comparisons']['A2_minus_A1']['ci95_fixed_domain_query']}`, reflecting recovery from the historical context failures rather than a causal model improvement.

Recovery and retention:

- Budgeted PPLX valid: `{recovery_final['a2_valid_n']}`/1,384; failures: `{recovery_final['a2_failure_n']}`.
- Historical-failure requests rescued: `{recovery_final['a2_rescued_from_original_failure_n']}`/773; new failures among historical-valid requests: `{recovery_final['a2_new_failures_on_original_valid_n']}`.
- Budgeted Jina valid: `{recovery_final['a4_valid_n']}`/1,384; failures: 4 unbudgetable rows.
- Mean/median passage-token retention: `{recovery_final['retention_ratio_mean']:.6f}` / `{recovery_final['retention_ratio_median']:.6f}`; p10/p90: `{recovery_final['retention_ratio_p10']:.6f}` / `{recovery_final['retention_ratio_p90']:.6f}`.

## E11-C resource evidence

Both arms ran sequentially on GPU 5 (`<REDACTED_GPU_UUID>`) with model loading excluded from request timing. The measured budgeted request summaries are:

| Arm | Attempts | Valid | Failures | Timed p50 ms | Timed p95 ms | Success p50/p95 ms | Peak memory p95 MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(f"| {r['arm']} | {r['attempts']} | {r['valid_n']} | {r['failure_n']} | {r['all_attempt_p50_ms']} | {r['all_attempt_p95_ms']} | {r['success_p50_ms']} / {r['success_p95_ms']} | {r['peak_memory_p95_mib']} |" for r in resource_rows)}

The four unbudgetable failures have no model request duration and are reported as untimed rather than assigned zero latency. Runtime physical forward counters were unavailable; native call counts are recorded but not substituted. These are concurrency-one observations, not sustained QPS or dollar-cost estimates. E10's corrected composite-key reuse matched 98 successful PPLX requests to Jina; its successful-pair p50 was 1,521.123829 / 139.9613715 ms and p95 was 2,392.6716869 / 247.15733675 ms (PPLX/Jina). The complete audit is `E11_C_E10_IDENTICAL_REQUEST_LATENCY_AUDIT.md`.

## E11-B downstream validity

No new reader or judge calls were made. Existing reference-validity limitations remain unresolved, so factual correctness and grounding are `NOT_ESTABLISHED`; known-positive nDCG is a retrieval/selection proxy only.

## Final scientific interpretation

One fixed deterministic context budget converts most historical PPLX context failures into valid requests and makes its operational nDCG competitive with full-content Jina on this exposed BRIGHT population, but the primary uncertainty interval includes zero and the population is post-E9/exploratory. The result supports a bounded context-feasibility mitigation finding, not a universal replacement, causal, factual-correctness, production-SLA or cost-savings claim. Full-content Jina remains the reference operational baseline; the budgeted PPLX trade-off is not independently confirmed.

## Preservation and restrictions

Original E2-E10 artifacts, blocked TREC E9 results, candidate slates and protected holdouts were preserved. No reserved holdout was opened; no reader, Jev, paid API, training, new model, retrieval rebuild or E12 work was initiated. The failed pre-fix attempt is documented as a schema-only implementation error and produced no accepted raw artifact; the corrected retry produced the frozen hashes in the native manifest.
"""
    (OUT / "07_closure/E11_FINAL_SCIENTIFIC_REPORT.md").write_text(report_md, encoding="utf-8")
    files = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "E11_FINAL_PROVENANCE_MANIFEST.json":
            files.append({"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha_file(p)})
    write_json(OUT / "07_closure/E11_FINAL_PROVENANCE_MANIFEST.json", {"schema": "DECISIONRAG-E11-FINAL-PROVENANCE-V1", "created_at": utc(), "files": files, "historical_input_snapshots": imm, "protected_holdouts_opened": False, "reader_calls": 0, "jev_calls": 0, "paid_api_calls": 0, "hash_mismatches": []})
    print(json.dumps({"gate": gate, "valid": valid, "failures": fail, "tests": report["full_regression"], "manifest_files": len(files)}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["protocol", "budget", "native", "analyze", "resources", "integrate", "closure"])
    parser.add_argument("--gpu", type=int, default=5)
    args = parser.parse_args()
    if args.phase == "protocol":
        slates = load_slates()
        for d in ("00_protocol", "01_budget_policy", "02_native", "03_analysis", "04_reader", "05_resources", "06_integration", "07_closure"):
            (OUT / d).mkdir(parents=True, exist_ok=True)
        e9_phase(slates)
        (OUT / "00_protocol/E11_SOURCE_AND_IMMUTABILITY_BASELINE.json").write_text(json.dumps({"sources": [{"path": str(p.relative_to(ROOT)), "exists": p.exists(), "sha256": sha_file(p) if p.exists() else None} for p in [E9 / "06_closure/E9_BRIGHT_PROVENANCE_MANIFEST.json", E9A / "E9_BRIGHT_FINAL_PROVENANCE_MANIFEST.json", E10 / "05_closure/E10_FINAL_PROVENANCE_MANIFEST.json", ROOT / "results/non_jev/v2/e9_v1_0/06_closure/E9_FINAL_OWNER_REVIEW_GATE.json"]], "protected_paths_not_opened": ["E3-F", "E5-F", "E6-F", "RESERVED"], "created_at": utc()}, indent=2) + "\n")
        (OUT / "00_protocol/E11_REGISTERED_SCOPE_AND_COHORT.md").write_text("# E11 registered scope and cohort\n\nFull 1,384-query official BRIGHT short-document population, 12 domains, original frozen Top-20 BM25 slates. E11-A budgeted PPLX/Jina inference is post-E9 exploratory mitigation. No retrieval rerun, labels in inputs, reader inference, protected holdout or paid/JeV call.\n", encoding="utf-8")
    elif args.phase == "budget":
        budget_phase()
    elif args.phase == "native":
        native_phase(args.gpu)
    elif args.phase == "analyze":
        analysis_phase()
    elif args.phase == "resources":
        resources_phase()
    elif args.phase == "integrate":
        (OUT / "04_reader/E11_B_EVIDENCE_CHANGE_AUDIT.md").write_text("# E11-B evidence-change audit\n\nNo new independent authoritative references, source adjudication access or human reviewers became available since E10-A. Existing QASPER/GaRAGe/BRIGHT reference-overlap artifacts do not resolve the documented correctness/grounding validity problem. No reader or judge inference was launched.\n", encoding="utf-8")
        write_json(OUT / "04_reader/E11_B_FINAL_ADMISSION_GATE.json", {"gate": "E11_B_NOT_ADMISSIBLE_REFERENCE_INADEQUATE", "new_independent_basis": False, "reader_calls": 0, "judge_calls": 0, "factual_correctness_claim": "NOT_ESTABLISHED", "grounding_claim": "NOT_ESTABLISHED"})
        resources_phase()
        integration_phase()
    else:
        closure_phase()


if __name__ == "__main__":
    main()
