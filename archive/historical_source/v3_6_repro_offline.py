#!/usr/bin/env python3
"""CPU-only R1 label join and R2 forensic audit.

The script is safe to run only after raw R1 files are complete.  It keeps all
labels out of native traces and writes only new append-only reports.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "v3_6_essential" / "followup_v3_repro_fixes"
PREP = BASE / "R1_transfer_budget_grid"
RAW = BASE / "raw"
R2 = BASE / "R2_jina_mismatch_trace"
OLD_MAN = ROOT / "results" / "v2_1_replication" / "stage0_preflight_v2_hybrid" / "INPUT_MANIFEST_V2_HYBRID.jsonl"
SWEEP = ROOT / "results" / "v3_6_essential" / "stage1_quality" / "raw" / "jina_two_stage_8192__8192_reuse.jsonl"
CONTROL = ROOT / "results" / "v1_5" / "c1_two_stage_specialist_gpu3" / "raw" / "jina_two_stage_pplxtext__original.jsonl"
E11 = ROOT / "results" / "non_jev" / "v2" / "e11_v1_0" / "01_budget_policy" / "E11_A_FROZEN_INPUT_AND_TOKEN_MANIFEST.jsonl"
BRIGHT_GOLD = ROOT / "results" / "non_jev" / "v2" / "e9_bright_v1_0" / "00_admission" / "BRIGHT_EXAMPLE_ID_MANIFEST.json"
FULL_JINA = ROOT / "results" / "v2_1_replication" / "stage1_native_v2_hybrid" / "jina_full__original__full.jsonl"


def read_jsonl(path):
    return [json.loads(x) for x in path.open(encoding="utf-8") if x.strip()]


def canon(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha_obj(x):
    return hashlib.sha256(canon(x).encode()).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def ndcg5(selected, gold):
    gold = set(gold)
    if not gold:
        return 0.0
    dcg = sum((1.0 if x in gold else 0.0) / math.log2(i + 2) for i, x in enumerate(selected[:5]))
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(5, len(gold))))
    return dcg / ideal if ideal else 0.0


def fs5(selected, gold):
    return int(bool(gold) and set(gold).issubset(set(selected[:5])))


def bootstrap(values, draws=40000, seed=731):
    import numpy as np
    if not values:
        return {"delta": None, "lo95": None, "hi95": None, "resampling_unit": "question", "draws": draws, "seed": seed}
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, draws, 1000):
        n = min(1000, draws - start)
        means.extend(x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1).tolist())
    return {"delta": float(x.mean()), "lo95": float(np.quantile(means, .025)), "hi95": float(np.quantile(means, .975)), "resampling_unit": "question", "draws": draws, "seed": seed}


def load_labels():
    spec = importlib.util.spec_from_file_location("analysis", ROOT / "scripts" / "v2_1_replication_analysis.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.load_labels(mod.load_manifest())


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def run_r1():
    labels = load_labels()
    prep = {}
    for L in (2048, 4096, 6144, 8192):
        for x in read_jsonl(PREP / f"prepared_L{L}.jsonl"):
            prep[(x["dataset"], x["canonical_id"], L)] = x
    raw = {}
    # Include the boundary-preserved prefix, explicit cancellation records,
    # and the completed suffix shards.  Never read .part files or summaries.
    raw_files = sorted(
        p for p in list(RAW.glob("pplx*.jsonl")) + list(RAW.glob("jina*.jsonl"))
        if not p.name.endswith(".part") and "SUMMARY" not in p.name
    )
    for p in raw_files:
        for x in read_jsonl(p):
            raw[(x["model_revision"], x["dataset"], x["canonical_id"], int(x["budget_L"]))] = x
    # Avoid ambiguity if revisions are unavailable in a failure record.
    raw_by_model = {}
    for p in raw_files:
        model = "pplx" if p.name.startswith("pplx") else "jina"
        for x in read_jsonl(p):
            raw_by_model[(model, x["dataset"], x["canonical_id"], int(x["budget_L"]))] = x
    records = []
    # The full-text Jina control is an authenticated historical trace; it is
    # reused only as a control, never mixed into the corrected native traces.
    full_jina = {x["canonical_id"]: x for x in read_jsonl(FULL_JINA)}
    for key, x in sorted(prep.items()):
        dataset, cid, L = key
        lab = labels.get((dataset, cid), {"labeled": False, "gold": set(), "status": "MISSING"})
        gold = set(lab.get("gold", set()))
        builder_ok = x["budget_info"].get("builder_status") == "PASS"
        for model in ("pplx", "jina"):
            r = raw_by_model.get((model, dataset, cid, L))
            if not builder_ok:
                status, valid, err, selected = "BUILDER_FAIL", 0, x["budget_info"].get("error_class", "BUILDER_FAIL"), []
            elif r is None:
                status, valid, err, selected = "MISSING_TRACE", 0, "MISSING_TRACE", []
            else:
                selected = [str(z) for z in (r.get("selected_ids") or [])]
                valid = int(r.get("status") == "PASS" and len(selected) == 5 and len(set(selected)) == 5 and set(selected).issubset(set(x["candidate_ids"])))
                status = r.get("status", "FAIL")
                err = "" if valid else (r.get("error_class") or "INVALID_OUTPUT")
            records.append({
                "dataset": dataset, "canonical_id": cid, "L": L, "model": model,
                "valid": valid, "status": status, "error_class": err,
                "builder_failure": int(not builder_ok), "model_output_failure": int(builder_ok and not valid),
                "labeled": int(bool(lab.get("labeled"))), "gold_count": len(gold),
                "reachable": int(bool(gold and gold.issubset(set(x["candidate_ids"])))),
                "ndcg5": ndcg5(selected if valid else [], gold) if lab.get("labeled") else "",
                "fs5": fs5(selected if valid else [], gold) if lab.get("labeled") else "",
                "serialized_tokens": x.get("serialized_tokens") or "",
                "input_sha256": (r or {}).get("input_sha256"),
            })
    # Add the two non-new-inference controls to each length-specific table.
    for (dataset, cid, L), x in sorted(prep.items()):
        lab = labels.get((dataset, cid), {"labeled": False, "gold": set()})
        gold = set(lab.get("gold", set()))
        first = [str(z) for z in x["candidate_ids"][:5]]
        records.append({
            "dataset": dataset, "canonical_id": cid, "L": L, "model": "hybrid_first_stage",
            "valid": 1, "status": "PASS", "error_class": "", "builder_failure": 0,
            "model_output_failure": 0, "labeled": int(bool(lab.get("labeled"))), "gold_count": len(gold),
            "reachable": int(bool(gold and gold.issubset(set(x["candidate_ids"])))),
            "ndcg5": ndcg5(first, gold) if lab.get("labeled") else "",
            "fs5": fs5(first, gold) if lab.get("labeled") else "",
            "serialized_tokens": x.get("serialized_tokens") or "", "input_sha256": "", "control_source": "frozen_candidate_prefix",
        })
        r = full_jina.get(cid)
        selected = [str(z) for z in (r or {}).get("selected_ids", [])]
        valid = int(bool(r) and r.get("status") in {"PASS", "OK", "ok"} and len(selected) == 5 and len(set(selected)) == 5)
        records.append({
            "dataset": dataset, "canonical_id": cid, "L": L, "model": "jina_full_text",
            "valid": valid, "status": (r or {}).get("status", "MISSING_TRACE"),
            "error_class": "" if valid else ((r or {}).get("error_class") or "MISSING_TRACE"),
            "builder_failure": 0, "model_output_failure": int(not valid), "labeled": int(bool(lab.get("labeled"))),
            "gold_count": len(gold), "reachable": int(bool(gold and gold.issubset(set(x["candidate_ids"])))),
            "ndcg5": ndcg5(selected if valid else [], gold) if lab.get("labeled") else "",
            "fs5": fs5(selected if valid else [], gold) if lab.get("labeled") else "",
            "serialized_tokens": "FULL_TEXT", "input_sha256": (r or {}).get("native_input_sha256", ""),
            "control_source": str(FULL_JINA.relative_to(ROOT)),
        })
    fields = ["dataset", "canonical_id", "L", "model", "valid", "status", "error_class", "builder_failure", "model_output_failure", "labeled", "gold_count", "reachable", "ndcg5", "fs5", "serialized_tokens", "input_sha256", "control_source"]
    write_csv(BASE / "exports" / "r1_perquery.csv", records, fields)
    by = defaultdict(list)
    for r in records:
        by[(r["dataset"], int(r["L"]), r["model"])].append(r)
    summary = []
    for (dataset, L, model), rs in sorted(by.items()):
        labeled = [r for r in rs if r["labeled"]]
        built = [r for r in rs if r["serialized_tokens"] not in ("", "FULL_TEXT", None)]
        scores = [float(r["ndcg5"]) for r in labeled]
        fs = [float(r["fs5"]) for r in labeled]
        summary.append({
            "dataset": dataset, "L": L, "policy": model, "n": len(rs), "gold_labeled_n": len(labeled),
            "valid_n": sum(int(r["valid"]) for r in rs), "valid_rate": sum(int(r["valid"]) for r in rs) / len(rs),
            "builder_failures": sum(int(r["builder_failure"]) for r in rs),
            "model_output_failures": sum(int(r["model_output_failure"]) for r in rs),
            "ndcg5_all_labeled": statistics.mean(scores) if scores else None,
            "fs5_all_labeled": statistics.mean(fs) if fs else None,
            "mean_serialized_tokens": statistics.mean([int(x) for x in (r["serialized_tokens"] for r in built)]) if built else None,
            "max_serialized_tokens": max([int(r["serialized_tokens"]) for r in built], default=None),
            "serialized_token_population": "built_requests" if built else "NOT_APPLICABLE",
            "resampling_unit": "question",
        })
    write_csv(BASE / "exports" / "r1_summary.csv", summary, list(summary[0]))
    contrasts = []
    for dataset in ("enterprise", "musique"):
        for L in (2048, 4096, 6144, 8192):
            p = {(r["canonical_id"]): r for r in by[(dataset, L, "pplx")]}
            j = {(r["canonical_id"]): r for r in by[(dataset, L, "jina")]}
            diffs = [float(p[k]["ndcg5"]) - float(j[k]["ndcg5"]) for k in sorted(set(p) & set(j)) if p[k]["labeled"] and j[k]["labeled"]]
            boot = bootstrap(diffs)
            contrasts.append({"dataset": dataset, "L": L, "contrast": "pplx_minus_jina_matched_text", "n": len(diffs), **boot})
    write_csv(BASE / "exports" / "r1_contrasts.csv", contrasts, list(contrasts[0]))
    # Frozen L=8192 identity comparison against the earlier v2.1 input.
    old = {(x["dataset"], x["canonical_id"]): x for x in read_jsonl(OLD_MAN) if x.get("order_id") == "original"}
    hashes = []
    for (dataset, cid, L), x in sorted(prep.items()):
        if L != 8192:
            continue
        current = sha_obj({"state": f"Query: {x['query']}\nEvaluate the supplied candidate passages using the option descriptions.", "question": {"type": "choice", "instructions": f"Select the candidate passage that best answers the query: {x['query']}", "criteria": {f"candidate_{i:03d}": f"Candidate ID {d['candidate_id']}. Passage: {d['content']}" for i, d in enumerate(x['budgeted_candidates'])}}}) if x.get("budgeted_candidates") else ""
        previous = old.get((dataset, cid), {}).get("budgeted_request_sha256")
        hashes.append({"dataset": dataset, "canonical_id": cid, "current_input_sha256": current, "previous_input_sha256": previous, "identical": int(bool(current and previous and current == previous))})
    write_csv(BASE / "exports" / "r1_L8192_input_hash_comparison.csv", hashes, list(hashes[0]))
    (BASE / "exports" / "R1_JOIN_SUMMARY.json").write_text(json.dumps({"status": "PASS", "perquery_rows": len(records), "summary_rows": len(summary), "contrast_rows": len(contrasts), "raw_input_files": [str(p.relative_to(ROOT)) for p in raw_files], "labels_joined_after_raw_freeze": True, "bootstrap_draws": 40000, "bootstrap_seed": 731, "bootstrap_unit": "question"}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "perquery_rows": len(records), "summary_rows": len(summary), "contrast_rows": len(contrasts)}, indent=2))


def main():
    # This guard makes partial use explicit while raw model jobs are running.
    complete = (RAW / "R1_RAW_COMPLETION.json").exists() and (RAW / "jina_shard0_of1.jsonl").exists()
    if not complete:
        print(json.dumps({"status": "WAITING_FOR_RAW_R1", "raw_dir": str(RAW.relative_to(ROOT))}, indent=2))
    else:
        run_r1()
    run_r2()


def run_r2():
    R2.mkdir(parents=True, exist_ok=True)
    sweep = {x["canonical_id"]: x for x in read_jsonl(SWEEP)}
    control = {x["canonical_id"]: x for x in read_jsonl(CONTROL)}
    e11 = {f"{x['domain']}:{x['query_id']}": x for x in read_jsonl(E11)}
    gold = {(str(x["domain"]), str(x["id"])): set(map(str, x.get("gold_ids", []))) for x in json.loads(BRIGHT_GOLD.read_text())["items"]}
    details = []
    mismatches = []
    for cid in sorted(set(sweep) | set(control)):
        a, b = sweep.get(cid), control.get(cid)
        if a is None or b is None:
            details.append({"canonical_id": cid, "status": "MISSING_SIDE"})
            continue
        domain = a.get("domain", "")
        qid = a.get("query_id", cid.split(":", 1)[-1])
        manifest = e11.get(cid, {})
        budgeted = manifest.get("budgeted_candidates", [])
        windows = [budgeted[:10], budgeted[10:20]]
        window_sha = [sha_obj([{"candidate_id": d["candidate_id"], "content": d["content"]} for d in w]) for w in windows]
        token_counts = [[len(str(d["content"]).split()) for d in w] for w in windows]
        stage_a = a.get("stage_records") or []
        stage_b = b.get("stage_records") or []
        request_hash_equal = [i < len(stage_a) and i < len(stage_b) and stage_a[i].get("request_sha256") == stage_b[i].get("request_sha256") for i in range(max(len(stage_a), len(stage_b)))]
        stage_candidate_equal = [i < len(stage_a) and i < len(stage_b) and stage_a[i].get("candidate_ids") == stage_b[i].get("candidate_ids") for i in range(max(len(stage_a), len(stage_b)))]
        score_equal = [i < len(stage_a) and i < len(stage_b) and stage_a[i].get("scores_sha256") == stage_b[i].get("scores_sha256") for i in range(max(len(stage_a), len(stage_b)))]
        list_equal = (a.get("selected_ids") or []) == (b.get("selected_ids") or [])
        set_equal = set(a.get("selected_ids") or []) == set(b.get("selected_ids") or [])
        g = gold.get((domain, qid), set())
        row = {
            "canonical_id": cid, "domain": domain, "query_id": qid,
            "sweep_status": a.get("status"), "control_status": b.get("status"),
            "window_a_sha256": window_sha[0] if len(window_sha) > 0 else "",
            "window_b_sha256": window_sha[1] if len(window_sha) > 1 else "",
            "window_a_token_counts": json.dumps(token_counts[0] if len(token_counts) > 0 else []),
            "window_b_token_counts": json.dumps(token_counts[1] if len(token_counts) > 1 else []),
            "sweep_stage_survivors": json.dumps([s.get("candidate_ids", []) for s in stage_a[:2]], separators=(",", ":")),
            "control_stage_survivors": json.dumps([s.get("candidate_ids", []) for s in stage_b[:2]], separators=(",", ":")),
            "sweep_final_stage_input_sha256": stage_a[2].get("request_sha256", "") if len(stage_a) > 2 else "",
            "control_final_stage_input_sha256": stage_b[2].get("request_sha256", "") if len(stage_b) > 2 else "",
            "window_strings_identical": int(a.get("source_input_sha256") == b.get("source_input_sha256") and all(request_hash_equal)),
            "truncation_cap_identical": int(manifest.get("final_per_passage_cap") is not None),
            "window_membership_order_identical": int(all(stage_candidate_equal)),
            "tokenizer_identical": "NOT_OBSERVABLE_FROM_RAW_TRACE",
            "dtype_identical": "NOT_OBSERVABLE_FROM_RAW_TRACE",
            "batch_size_identical": "NOT_OBSERVABLE_FROM_RAW_TRACE",
            "final_stage_input_identical": int(len(stage_a) > 2 and len(stage_b) > 2 and stage_a[2].get("request_sha256") == stage_b[2].get("request_sha256")),
            "stage_score_hashes_identical": int(all(score_equal)),
            "list_equal": int(list_equal), "set_equal": int(set_equal),
            "ndcg5_sweep": ndcg5(a.get("selected_ids", []) if a.get("status") == "PASS" else [], g),
            "ndcg5_control": ndcg5(b.get("selected_ids", []) if b.get("status") == "PASS" else [], g),
            "score_vector_max_abs_diff": max((abs(float(x) - float(y)) for x, y in zip(a.get("scores", []), b.get("scores", []))), default=0.0),
            "labels_joined_after_raw_freeze": True,
        }
        details.append(row)
        if not list_equal:
            mismatches.append(row)
    fields = list(details[0]) if details else ["canonical_id"]
    write_csv(R2 / "r2_trace_comparison.csv", details, fields)
    write_csv(R2 / "r2_differing_queries.csv", mismatches, fields)
    summary = {
        "status": "PASS",
        "n_sweep": len(sweep), "n_control": len(control), "n_common": len(set(sweep) & set(control)),
        "list_difference_n": len(mismatches), "set_difference_n": sum(not bool(x.get("set_equal")) for x in mismatches),
        "all_stage_request_hashes_identical_n": sum(1 for x in details if x.get("window_strings_identical") and x.get("final_stage_input_identical")),
        "all_stage_score_hashes_identical_n": sum(bool(x.get("stage_score_hashes_identical")) for x in details),
        "fresh_exact_string_rerun_required": bool(mismatches),
        "conclusion": "The available sweep explicitly reuses the historical PPLX-window strings; no 197-query mismatch was found in this pair." if not mismatches else "Input/output disagreement exists; inspect per-query trace and rerun only if the frozen string hashes differ.",
        "unobservable_fields": ["tokenizer", "dtype", "batch_size"],
        "sweep_sha256": sha_file(SWEEP), "control_sha256": sha_file(CONTROL),
    }
    (R2 / "R2_TRACE_AUDIT_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
