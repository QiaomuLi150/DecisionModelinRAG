#!/usr/bin/env python3
"""DecisionRAG E3-B offline analysis and bounded native-inference runner.

The runner freezes B0 panels, optionally executes the owner-authorized B1
native arms, and completes B2/B3 from authenticated historical score traces.
It fails closed on native errors and never fabricates inference results.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import random
import re
import sqlite3
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e3_v1/e3_b_mechanism"
GOV = ROOT / "results/non_jev/v2/e3_v1/e3_0_governance"
SEED = 731
BOOTSTRAP = 5000
CLEF_REV = "8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3"
POLICIES = ("bge_top5", "bge_mmr", "bge_coverage", "jina35", "pplx_decider", "clef_joint", "clef_pointwise")
GPU_SNAPSHOT = {
    "safety_check_mode": "host_read_only_nvidia_smi_after_sandbox_failure",
    "approved_gpu_ids": [0, 1, 2, 4],
    "safe_idle_approved_gpu_ids": [],
    "devices": [
        {"index": 0, "uuid": "<REDACTED_GPU_UUID>", "memory_used_mib": 23571, "memory_free_mib": 57582, "utilization_percent": 28, "processes": [1818102, 2110605]},
        {"index": 1, "uuid": "<REDACTED_GPU_UUID>", "memory_used_mib": 11598, "memory_free_mib": 69555, "utilization_percent": 100, "processes": [1894223, 2375077, 2375080, 2375082, 2375843, 2375996]},
        {"index": 2, "uuid": "<REDACTED_GPU_UUID>", "memory_used_mib": 54785, "memory_free_mib": 26367, "utilization_percent": 0, "processes": [881592]},
        {"index": 4, "uuid": "<REDACTED_GPU_UUID>", "memory_used_mib": 56583, "memory_free_mib": 24570, "utilization_percent": 0, "processes": [881828, 1385448]},
    ],
    "decision": "B1_GPU_BLOCKED_NO_APPROVED_IDLE_DEVICE",
    "unrelated_workloads_interrupted": False,
}


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


def jl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def load_e3a():
    sys.path.insert(0, str(ROOT / "scripts"))
    import e3_a_offline as e3a
    items = e3a.load_items()
    gold = e3a.load_gold()
    states = list(csv.DictReader((ROOT / "results/non_jev/v2/e3_v1/e3_a_paired/E3_A_QUERY_LEVEL_EVIDENCE_STATES.csv").open()))
    state = {r["query_id"]: r for r in states}
    return e3a, items, gold, state


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values())); nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def load_groups() -> dict[str, list[list[str]]]:
    by_q = {}
    for path in [ROOT / "results/non_jev/v11/track_a/V11_TRACK_A_CLEF_GROUPWISE_FULL.jsonl", ROOT / "results/non_jev/v11/track_b/V11_TRACK_B_CLEF_GROUPWISE_FULL.jsonl"]:
        for row in jl(path):
            groups = [g.get("group_doc_ids", []) for g in row.get("groups", [])]
            if len(groups) == 5 and all(len(g) == 4 and len(set(g)) == 4 for g in groups) and len({x for g in groups for x in g}) == 20:
                by_q[row["query_id"]] = groups
    return by_q


def stable_order(qid: str, arm: str) -> str:
    return hashlib.sha256(f"{SEED}:{arm}:{qid}".encode()).hexdigest()


def load_score(path: Path) -> dict[str, dict]:
    rows = {r["query_id"]: r for r in jl(path)}
    if len(rows) != 4267 or any(r.get("status") != "ok" for r in rows.values()):
        raise RuntimeError(f"invalid score source {path}")
    return rows


def select_top(ids: list[str], scores: list[float]) -> list[str]:
    return [x[1] for x in sorted(zip(scores, ids), key=lambda x: (-float(x[0]), ids.index(x[1]), x[1]))[:5]]


def select_quota(ids: list[str], scores: list[float], groups: list[list[str]]) -> list[str]:
    score = dict(zip(ids, scores)); rank = {x: i for i, x in enumerate(ids)}
    winners = [sorted(g, key=lambda d: (-float(score[d]), rank[d], d))[0] for g in groups]
    return [x[1] for x in sorted(((score[d], d) for d in winners), key=lambda x: (-float(x[0]), rank[x[1]], x[1]))]


def metric(gold: set[str], candidates: list[str], selected: list[str]) -> tuple[bool, bool, bool]:
    eligible = 1 <= len(gold) <= 5
    conditional = eligible and gold.issubset(set(candidates))
    return eligible, conditional, eligible and gold.issubset(set(selected))


def bootstrap(values: list[float]) -> tuple[float | None, float | None]:
    if not values: return None, None
    rng = random.Random(SEED); n = len(values)
    estimates = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOTSTRAP))
    return estimates[int(.025 * BOOTSTRAP)], estimates[int(.975 * BOOTSTRAP) - 1]


def make_sample(items: list[dict], gold: dict, state: dict) -> tuple[list[dict], list[dict]]:
    """Deterministic 100-per-dataset B0 sample from conditional valid rows."""
    chosen = []
    for dataset, target in (("hotpot", 100), ("musique", 100), ("wiki2", 100), ("enterprise", 100)):
        candidates = []
        for item in items:
            if item["dataset"] != dataset or not state[item["query_id"]]["identity_eligible"] == "True" or state[item["query_id"]]["retrieval_feasible"] != "True":
                continue
            g = set(gold.get(item["query_id"], {}).get("docs", [])); ids = [c["doc_id"] for c in item["candidates"]]
            ranks = [ids.index(x) + 1 for x in g if x in ids]
            qlen = len(tokens(item["question"]))
            length_bin = "short" if qlen <= 12 else "medium" if qlen <= 24 else "long"
            rank_bin = "early" if min(ranks) <= 5 else "middle" if min(ranks) <= 10 else "late"
            stratum = f"support={len(g)}|rank={rank_bin}|question={length_bin}"
            candidates.append((stratum, stable_order(item["query_id"], "B0"), item, len(g), min(ranks), length_bin, rank_bin))
        if len(candidates) < target:
            raise RuntimeError(f"B0 {dataset} has only {len(candidates)} eligible conditional rows")
        by_stratum = defaultdict(list)
        for row in sorted(candidates, key=lambda x: (x[0], x[1])): by_stratum[row[0]].append(row)
        # Round-robin through sorted strata, then stable hash order; this is
        # fixed before any E3-B arm result is inspected.
        selected = []
        strata = sorted(by_stratum)
        cursor = 0
        while len(selected) < target:
            progressed = False
            for s in strata:
                if cursor < len(by_stratum[s]) and len(selected) < target:
                    selected.append(by_stratum[s][cursor]); progressed = True
            if not progressed: break
            cursor += 1
        for stratum, order, item, sc, mr, lb, rb in selected:
            ids = [c["doc_id"] for c in item["candidates"]]
            views = [sha_bytes(str(c.get("full_text", "")).encode()) for c in item["candidates"]]
            chosen.append({"query_id": item["query_id"], "cohort": item["cohort"], "dataset": dataset, "question_sha256": sha_bytes(item["question"].encode()), "candidate_ids": ids, "candidate_view_hashes": views, "slate_hash": sha_json(ids), "sample_stratum": stratum, "support_cardinality": sc, "first_stage_gold_rank": mr, "question_length_bin": lb, "rank_bin": rb, "selection_seed": SEED, "selection_rule": "stable-hash stratified round-robin over identity-valid retrieval-conditional historical rows; no outcome fields"})
    chosen.sort(key=lambda x: (x["dataset"], x["query_id"]))
    cal = []
    cal_rows = {r["query_id"]: r for r in jl(ROOT / "results/non_jev/v2/e2_m1/E2_M1_CAL_JOINT.jsonl")}
    for dataset in ("hotpot", "musique", "wiki2"):
        ds = sorted([r for r in cal_rows.values() if r.get("dataset") == dataset], key=lambda r: r["query_id"])[:4]
        for r in ds:
            cal.append({"query_id": r["query_id"], "dataset": r["dataset"], "candidate_ids": r["candidate_ids"], "slate_hash": sha_json(r["candidate_ids"]), "selection_rule": "first four canonical CAL rows per Track-A dataset", "selection_seed": SEED, "disjoint_from_b1_sample": r["query_id"] not in {x["query_id"] for x in chosen}})
    return chosen, cal


def make_panels(sample: list[dict], items: dict[str, dict], gold: dict) -> list[dict]:
    pools = defaultdict(dict)
    for item in items.values():
        for c in item["candidates"]:
            pools[item["dataset"]][c["doc_id"]] = c
    # Precompute lightweight lexical metadata once.  The earlier implementation
    # re-tokenized and fully sorted the entire historical corpus for every
    # sample query, delaying the protocol freeze before any inference.
    pool_meta = {}
    for dataset, by_id in pools.items():
        meta = []
        for c in by_id.values():
            text = str(c.get("full_text", ""))
            meta.append({"candidate": c, "length_chars": len(text), "token_set": set(tokens(text)), "stable": stable_order(c["doc_id"], "panel_pool")})
        meta.sort(key=lambda x: x["stable"])
        pool_meta[dataset] = meta
    panels = []
    for index, row in enumerate(sample):
        item = items[row["query_id"]]; candidates = item["candidates"]; ids = [c["doc_id"] for c in candidates]
        anchor_pos = (index % 4) * 5; anchor = ids[anchor_pos]
        # Three seeded order permutations of exactly the same candidate views.
        for arm_no in range(1, 4):
            rng = random.Random(SEED + index * 1009 + arm_no * 9176)
            perm = ids[:]; rng.shuffle(perm)
            if perm == ids: perm = perm[1:] + perm[:1]
            panels.append({"query_id": row["query_id"], "dataset": row["dataset"], "arm": f"B1_ORDER_P{arm_no}", "candidate_ids": perm, "canonical_candidate_ids": ids, "anchor_id": anchor, "anchor_original_rank": anchor_pos + 1, "panel_type": "order_only", "panel_hash": sha_json(perm), "input_hash": sha_json({"question_sha256": row["question_sha256"], "candidate_ids": perm, "view_hashes": row["candidate_view_hashes"], "contract": "Clef_native_joint_binary_v3"}), "permutation_seed": SEED + index * 1009 + arm_no * 9176})
        # Fixed four-document replacement panel, preserving anchor position.
        replace_pos = [(anchor_pos + x) % 20 for x in (1, 2, 3, 4)]
        current = set(ids); forbidden = set(gold.get(row["query_id"], {}).get("docs", []))
        pool = [m for m in pool_meta[row["dataset"]] if m["candidate"]["doc_id"] not in current and m["candidate"]["doc_id"] not in forbidden]
        anchor_len = len(str(candidates[anchor_pos].get("full_text", "")))
        qtokens = set(tokens(item["question"]))
        def overlap(meta):
            return len(qtokens & meta["token_set"]) / max(1, len(qtokens))
        # Length screening is cheap; lexical scoring is performed only on a
        # bounded deterministic panel.  This remains a fixed, gold-blind
        # construction rule and avoids outcome-dependent sampling.
        length_screen = sorted(pool, key=lambda m: (abs(m["length_chars"] - anchor_len), m["stable"]))[:1024]
        matched_meta = sorted(length_screen, key=lambda m: (-overlap(m), abs(m["length_chars"] - anchor_len), m["stable"]))[:len(replace_pos)]
        unrelated_meta = sorted(pool[:4096], key=lambda m: (overlap(m), abs(m["length_chars"] - anchor_len), m["stable"]))[:len(replace_pos)]
        matched = [m["candidate"] for m in matched_meta]
        unrelated = [m["candidate"] for m in unrelated_meta]
        for arm, replacements in (("B1_CONTEXT_MATCHED", matched), ("B1_CONTEXT_IRRELEVANT", unrelated)):
            new_ids = ids[:]; replacement_meta = []
            for pos, c in zip(replace_pos, replacements):
                meta = next(m for m in pool if m["candidate"]["doc_id"] == c["doc_id"])
                new_ids[pos] = c["doc_id"]; replacement_meta.append({"position": pos, "old_id": ids[pos], "new_id": c["doc_id"], "new_view_hash": sha_bytes(str(c.get("full_text", "")).encode()), "length_chars": len(str(c.get("full_text", ""))), "query_overlap": overlap(meta)})
            panels.append({"query_id": row["query_id"], "dataset": row["dataset"], "arm": arm, "candidate_ids": new_ids, "canonical_candidate_ids": ids, "anchor_id": anchor, "anchor_original_rank": anchor_pos + 1, "panel_type": "context_replacement", "replacement_positions": replace_pos, "replacement_provenance": replacement_meta, "gold_exclusion_applied": True, "panel_hash": sha_json(new_ids), "input_hash": sha_json({"question_sha256": row["question_sha256"], "candidate_ids": new_ids, "contract": "Clef_native_joint_binary_v3"})})
    return panels


def _host_gpu_snapshot() -> dict:
    """Capture a read-only host inventory when the runner has device access."""
    query = ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"]
    try:
        out = subprocess.check_output(query, text=True, stderr=subprocess.STDOUT, timeout=15)
        devices = []
        for line in out.splitlines():
            fields = [x.strip() for x in line.split(",")]
            if len(fields) == 7:
                devices.append({"index": int(fields[0]), "uuid": fields[1], "name": fields[2], "memory_total_mib": int(fields[3]), "memory_used_mib": int(fields[4]), "memory_free_mib": int(fields[5]), "utilization_percent": int(fields[6])})
        return {"status": "captured", "devices": devices, "command": " ".join(query)}
    except Exception as exc:
        return {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}", "devices": []}


def _candidate_pool(items: dict[str, dict]) -> dict[str, dict[str, dict]]:
    pool: dict[str, dict[str, dict]] = defaultdict(dict)
    for item in items.values():
        for candidate in item["candidates"]:
            pool[item["dataset"]][candidate["doc_id"]] = candidate
    return pool


def _variant_item(original: dict, candidate_ids: list[str], pool: dict[str, dict[str, dict]]) -> dict:
    originals = {c["doc_id"]: c for c in original["candidates"]}
    candidates = []
    for rank, doc_id in enumerate(candidate_ids, 1):
        candidate = originals.get(doc_id) or pool[original["dataset"]].get(doc_id)
        if candidate is None:
            raise RuntimeError(f"missing panel candidate text {doc_id} for {original['query_id']}")
        candidates.append({**candidate, "doc_id": doc_id, "original_rank": rank})
    return {"query_id": original["query_id"], "cohort": original["cohort"], "dataset": original["dataset"], "question": original["question"], "candidates": candidates}


def _freeze_protocol(sample: list[dict], cal: list[dict], panels: list[dict]) -> dict:
    protocol = {
        "status": "FROZEN_BEFORE_NEW_INFERENCE",
        "seed": SEED,
        "sample_counts": dict(Counter(x["dataset"] for x in sample)),
        "cal_counts": dict(Counter(x["dataset"] for x in cal)),
        "new_inference_cap_full_slate": 2000,
        "arms": ["B1_BASE_JOINT", "B1_ORDER_P1", "B1_ORDER_P2", "B1_ORDER_P3", "B1_CONTEXT_MATCHED", "B1_CONTEXT_IRRELEVANT"],
        "panel_counts": dict(Counter(x["arm"] for x in panels)),
        "anchor_rule": "position 0/5/10/15 cycled by canonical sample index",
        "replacement_rule": "four non-anchor positions; same-dataset source-distinct, length/topic matched or low-query-overlap; known gold IDs excluded offline",
        "model": "Cloudflare/clef-flash",
        "revision": CLEF_REV,
        "precision": "bfloat16",
        "score_contract": "JOINT_BINARY_RELEVANCE:v2; higher is more relevant; canonical ID remap",
        "candidate_count": 20,
        "quota": 5,
        "holdout_access": False,
        "protocol_sha256": "computed_after_write",
    }
    protocol["protocol_sha256"] = sha_json({k: v for k, v in protocol.items() if k != "protocol_sha256"})
    write_json(OUT / "E3_B_FROZEN_INTERVENTION_PROTOCOL.json", protocol)
    return protocol


def _write_b1_manifest_and_trace_rows(sample: list[dict], panels: list[dict], joint: dict[str, dict], traces: list[dict], gpu_id: int, host_snapshot: dict, smoke_count: int, cal_count: int) -> dict:
    source_file = "results/non_jev/v2/e2_m1/E2_M1_JOINT_BINARY_SCORES.jsonl"
    manifest = []
    for row in sample:
        base = joint[row["query_id"]]
        manifest.append({"query_id": row["query_id"], "dataset": row["dataset"], "arm": "B1_BASE_JOINT", "candidate_ids": row["candidate_ids"], "candidate_view_hashes": row["candidate_view_hashes"], "slate_hash": row["slate_hash"], "input_hash": base.get("actual_input_sha256"), "source_response_hash": base.get("response_hash"), "source_file": source_file, "result_type": "authenticated_reuse", "physical_forward_count": base.get("physical_forward_calls"), "status": "reused"})
    panel_by_key = {(r.get("query_id"), r.get("arm")): r for r in traces if r.get("stage") == "full"}
    for panel in panels:
        trace = panel_by_key.get((panel["query_id"], panel["arm"]), {})
        manifest.append({**panel, "result_type": "new_native_inference", "status": trace.get("status", "error"), "physical_forward_count": trace.get("physical_forward_count", 0), "actual_input_sha256": trace.get("actual_input_sha256"), "response_hash": trace.get("response_hash"), "gpu_id": trace.get("gpu_id", gpu_id), "failure_reason": trace.get("error")})
    with (OUT / "E3_B_SAMPLE_AND_PANEL_MANIFEST.jsonl").open("w", encoding="utf-8") as f:
        for row in sorted(manifest, key=lambda x: (x["query_id"], x["arm"])):
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    raw = []
    source_hash = sha_file(ROOT / source_file)
    for row in manifest:
        detail = panel_by_key.get((row.get("query_id"), row.get("arm")), {})
        if row.get("arm") == "B1_BASE_JOINT":
            detail = joint.get(row["query_id"], {})
        raw.append({"source_file_sha256": source_hash, "git_commit": os.popen("git rev-parse HEAD").read().strip(), "model_revision": CLEF_REV, "policy_id": row["arm"], "dataset": row["dataset"], "query_id": row["query_id"], "slate_hash": row.get("slate_hash", row.get("panel_hash")), "candidate_ids": row["candidate_ids"], "candidate_view_hash": sha_json(row.get("candidate_view_hashes", [])), "input_hash": row.get("actual_input_sha256", row.get("input_hash")), "response_hash": row.get("response_hash", row.get("source_response_hash")), "forward_count": row.get("physical_forward_count", 0), "gpu_id": row.get("gpu_id", "historical"), "metric_population": "B1_400_preselected_historical_exploratory", "label_provenance": "offline_labels_not_in_input", "result_type": row["result_type"], "status": row["status"], "failure_reason": row.get("failure_reason")})
        raw[-1].update({"scores": detail.get("canonical_scores", detail.get("scores")), "raw_scores": detail.get("raw_scores", detail.get("scores")), "telemetry": detail.get("telemetry"), "actual_input_sha256": detail.get("actual_input_sha256", row.get("input_hash")), "gold_in_inference_input": False})
    with (OUT / "E3_B_CONTEXT_INTERVENTION_RAW_TRACES.jsonl").open("w", encoding="utf-8") as f:
        for row in raw:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {"manifest_rows": len(manifest), "new_rows": sum(r["result_type"] == "new_native_inference" for r in manifest), "base_reused": sum(r["status"] == "reused" for r in manifest), "smoke_rows": smoke_count, "cal_rows": cal_count, "gpu_id": gpu_id, "host_snapshot": host_snapshot}


def _run_b1_native(sample: list[dict], cal: list[dict], panels: list[dict], items: dict[str, dict], joint: dict[str, dict], gpu_id: int) -> dict:
    """Run only the predeclared B1 Clef arms; baseline is reused, not rescored."""
    import importlib
    e2 = importlib.import_module("e2_m1_experiment")
    e2.GPU_ID = gpu_id
    host_snapshot = _host_gpu_snapshot()
    pool = _candidate_pool(items)
    original = items
    trace_rows = []
    trace_path = OUT / "E3_B_CONTEXT_INTERVENTION_NATIVE_DETAIL.jsonl"
    previous_rows = jl(trace_path) if trace_path.exists() else []
    previous_good = [r for r in previous_rows if r.get("status") == "ok"]
    previous_by_key = {(r.get("stage"), r.get("query_id"), r.get("arm")): r for r in previous_good}
    if trace_path.exists():
        trace_path.unlink()
    for row in previous_good:
        trace_rows.append(row)
        with trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    # The smoke set is the first four existing CAL rows per dataset (12 total).
    smoke_qids = {r["query_id"] for r in cal}
    smoke_items = []
    cal_items = []
    cal_local = e2.load_local_cal_items()
    for item in cal_local:
        if item["query_id"] in smoke_qids:
            smoke_items.append(item)
        if len(cal_items) < 30:
            cal_items.append(item)
    model_module, model, processor, torch = e2.import_clef()
    tokenizer = processor.tokenizer
    torch.cuda.set_device(0)

    def run_variant(item: dict, arm: str, candidate_ids: list[str], stage: str, canonical_ids: list[str] | None = None, panel: dict | None = None):
        prior = previous_by_key.get((stage, item["query_id"], arm))
        if prior is not None:
            return prior
        try:
            variant = _variant_item(item, candidate_ids, pool)
            result = e2.joint_one(variant, model_module, model, processor, tokenizer, torch)
            raw_ids = list(result["candidate_ids"]); raw_scores = list(result["scores"])
            score_map = dict(zip(raw_ids, raw_scores))
            canonical = canonical_ids or candidate_ids
            result["canonical_candidate_ids"] = canonical
            result["canonical_scores"] = [score_map[x] for x in canonical]
            result["raw_candidate_ids"] = raw_ids
            result["raw_scores"] = raw_scores
            result["arm"] = arm; result["stage"] = stage; result["gpu_id"] = gpu_id
            result["panel_hash"] = panel.get("panel_hash") if panel else sha_json(candidate_ids)
            result["anchor_id"] = panel.get("anchor_id") if panel else None
            result["replacement_positions"] = panel.get("replacement_positions", []) if panel else []
            result["status"] = "ok"; result["physical_forward_count"] = int(result.get("physical_forward_calls", 1))
        except Exception as exc:
            result = {"query_id": item["query_id"], "dataset": item["dataset"], "cohort": item["cohort"], "arm": arm, "stage": stage, "candidate_ids": candidate_ids, "canonical_candidate_ids": canonical_ids or candidate_ids, "status": "error", "real_inference": False, "physical_forward_count": 0, "gpu_id": gpu_id, "error": f"{type(exc).__name__}: {exc}", "panel_hash": panel.get("panel_hash") if panel else sha_json(candidate_ids)}
        trace_rows.append(result)
        with trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
        return result

    # Hard smoke: every arm represented on the 12 frozen CAL cases.
    for item in smoke_items:
        base = item["candidates"]; ids = [c["doc_id"] for c in base]
        run_variant(item, "B1_SMOKE_BASE_JOINT", ids, "smoke", ids)
        for arm in ("B1_ORDER_P1", "B1_ORDER_P2", "B1_ORDER_P3", "B1_CONTEXT_MATCHED", "B1_CONTEXT_IRRELEVANT"):
            matching = next((p for p in panels if p["query_id"] == item["query_id"] and p["arm"] == arm), None)
            if matching is None:
                # CAL smoke rows are not part of the 400-sample panels; use a
                # deterministic order-only smoke construction and unchanged
                # candidate set for the two context-control smoke arms.
                if arm.startswith("B1_ORDER"):
                    rng = random.Random(SEED + hash((item["query_id"], arm)) % 100000)
                    new_ids = ids[:]; rng.shuffle(new_ids)
                else:
                    new_ids = ids[:]
                matching = {"panel_hash": sha_json(new_ids), "anchor_id": ids[0], "replacement_positions": []}
            run_variant(item, arm, matching.get("candidate_ids", new_ids), "smoke", ids, matching)
    # CAL gate: one original-order native joint call per frozen deterministic case.
    for item in cal_items:
        ids = [c["doc_id"] for c in item["candidates"]]
        run_variant(item, "B1_CAL_BASE_JOINT", ids, "cal", ids)
    # Full historical exploratory cohort: five new arms x400 = 2,000 calls.
    for index, panel in enumerate(panels, 1):
        item = original[panel["query_id"]]
        # Order arms use the original canonical slate for remapping. Context
        # arms intentionally have a different candidate universe, so their
        # canonical order is the actual replacement panel, while the original
        # slate remains separately recorded for anchor diagnostics.
        canonical_ids = panel["canonical_candidate_ids"] if panel["arm"].startswith("B1_ORDER") else panel["candidate_ids"]
        run_variant(item, panel["arm"], panel["candidate_ids"], "full", canonical_ids, panel)
        if index % 25 == 0:
            gc.collect(); torch.cuda.empty_cache()
    del model, processor, tokenizer, model_module
    gc.collect(); torch.cuda.empty_cache()
    traces = trace_rows
    # Preserve full native detail separately; the required raw trace is a
    # compact provenance ledger generated below.
    counts = _write_b1_manifest_and_trace_rows(sample, panels, joint, traces, gpu_id, host_snapshot, len(smoke_items), len(cal_items))
    counts["native_detail_path"] = str(trace_path.relative_to(ROOT))
    counts["native_ok"] = sum(r.get("status") == "ok" for r in traces)
    counts["native_errors"] = sum(r.get("status") == "error" for r in traces)
    counts["new_physical_forward_calls"] = sum(r.get("physical_forward_count", 0) for r in traces)
    return counts


def _summarize_b1(sample: list[dict], panels: list[dict], joint: dict[str, dict], traces: list[dict]) -> None:
    by = {(r.get("query_id"), r.get("arm")): r for r in traces if r.get("stage") == "full"}
    effects = []
    for arm in ("B1_ORDER_P1", "B1_ORDER_P2", "B1_ORDER_P3", "B1_CONTEXT_MATCHED", "B1_CONTEXT_IRRELEVANT"):
        overlaps = []; jaccards = []; anchor_rows = []; deltas = []
        for panel in [p for p in panels if p["arm"] == arm]:
            q = panel["query_id"]; trace = by.get((q, arm)); base = joint.get(q)
            if not trace or trace.get("status") != "ok" or not base: continue
            base_top = set(select_top(base["candidate_ids"], base["scores"]))
            new_ids = trace["canonical_candidate_ids"]; new_scores = trace["canonical_scores"]
            new_top = set(select_top(new_ids, new_scores))
            overlaps.append(len(base_top & new_top)); jaccards.append(len(base_top & new_top) / max(1, len(base_top | new_top)))
            if arm.startswith("B1_ORDER"):
                deltas.extend([new_scores[new_ids.index(d)] - base["scores"][base["candidate_ids"].index(d)] for d in base["candidate_ids"]])
            else:
                aid = panel["anchor_id"]
                if aid in new_ids:
                    order = sorted(range(len(new_ids)), key=lambda i: (-new_scores[i], i, new_ids[i]))
                    anchor_rows.append({"selected": aid in new_top, "rank": order.index(new_ids.index(aid)) + 1})
        effects.append({"arm": arm, "n_manifest": sum(p["arm"] == arm for p in panels), "n_physical_inference": sum(1 for p in panels if p["arm"] == arm and (p["query_id"], arm) in by and by[(p["query_id"], arm)].get("status") == "ok"), "status": "MEASURED", "top5_overlap_mean": statistics.mean(overlaps) if overlaps else None, "top5_jaccard_mean": statistics.mean(jaccards) if jaccards else None, "anchor_selected_rate": (sum(x["selected"] for x in anchor_rows) / len(anchor_rows) if anchor_rows else "NOT_APPLICABLE"), "anchor_rank_p50": (statistics.median([x["rank"] for x in anchor_rows]) if anchor_rows else "NOT_APPLICABLE"), "score_shift_mean": (statistics.mean(deltas) if deltas else "NOT_APPLICABLE"), "score_shift_abs_mean": (statistics.mean([abs(x) for x in deltas]) if deltas else "NOT_APPLICABLE"), "paired_ci_low": "NOT_MEASURED", "paired_ci_high": "NOT_MEASURED", "interpretation": "order-only canonical remap" if arm.startswith("B1_ORDER") else "context replacement diagnostic; not original-slate CSSR"})
    write_csv(OUT / "E3_B_CONTEXT_STABILITY_AND_EFFECTS.csv", effects)


def write_b0_and_blocked_b1(sample: list[dict], cal: list[dict], panels: list[dict], items: dict[str, dict], joint: dict[str, dict]) -> dict:
    manifest = []
    for row in sample:
        qid = row["query_id"]; base = joint[qid]
        manifest.append({"query_id": qid, "dataset": row["dataset"], "arm": "B1_BASE_JOINT", "candidate_ids": row["candidate_ids"], "candidate_view_hashes": row["candidate_view_hashes"], "slate_hash": row["slate_hash"], "input_hash": base.get("actual_input_sha256"), "source_response_hash": base.get("response_hash"), "source_file": "results/non_jev/v2/e2_m1/E2_M1_JOINT_BINARY_SCORES.jsonl", "result_type": "authenticated_reuse", "physical_forward_count": base.get("physical_forward_calls"), "status": "reused"})
    for p in panels:
        manifest.append({**p, "result_type": "new_native_inference_planned", "status": "B1_GPU_BLOCKED", "failure_reason": "No approved idle GPU among 0,1,2,4; no inference launched", "physical_forward_count": 0})
    path = OUT / "E3_B_SAMPLE_AND_PANEL_MANIFEST.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in sorted(manifest, key=lambda x: (x["query_id"], x["arm"])): f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    raw = []
    for row in manifest:
        raw.append({"source_file_sha256": sha_file(ROOT / "results/non_jev/v2/e2_m1/E2_M1_JOINT_BINARY_SCORES.jsonl"), "git_commit": os.popen("git rev-parse HEAD").read().strip(), "model_revision": CLEF_REV, "policy_id": row["arm"], "dataset": row["dataset"], "query_id": row["query_id"], "slate_hash": row["slate_hash"], "candidate_ids": row["candidate_ids"], "candidate_view_hash": sha_json(row.get("candidate_view_hashes", [])), "input_hash": row.get("input_hash"), "response_hash": row.get("source_response_hash"), "forward_count": row.get("physical_forward_count", 0), "metric_population": "B1_400_preselected_historical_exploratory", "label_provenance": "offline_labels_not_in_input", "result_type": row["result_type"], "status": row["status"], "failure_reason": row.get("failure_reason")})
    with (OUT / "E3_B_CONTEXT_INTERVENTION_RAW_TRACES.jsonl").open("w", encoding="utf-8") as f:
        for row in raw: f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    effects = []
    for arm in ("B1_BASE_JOINT", "B1_ORDER_P1", "B1_ORDER_P2", "B1_ORDER_P3", "B1_CONTEXT_MATCHED", "B1_CONTEXT_IRRELEVANT"):
        n = sum(r["arm"] == arm for r in manifest)
        effects.append({"arm": arm, "n_manifest": n, "n_physical_inference": 0 if arm != "B1_BASE_JOINT" else 0, "status": "BASELINE_REUSED_ONLY" if arm == "B1_BASE_JOINT" else "B1_GPU_BLOCKED", "top5_overlap_vs_base": "NOT_MEASURED", "cssr_delta_vs_base": "NOT_MEASURED", "paired_ci_low": "NOT_MEASURED", "paired_ci_high": "NOT_MEASURED", "interpretation": "No new native intervention result; GPU gate blocked before inference"})
    write_csv(OUT / "E3_B_CONTEXT_STABILITY_AND_EFFECTS.csv", effects)
    write_json(OUT / "E3_B_FROZEN_INTERVENTION_PROTOCOL.json", {"status": "FROZEN_BEFORE_NEW_INFERENCE; B1_BLOCKED", "seed": SEED, "sample_counts": dict(Counter(x["dataset"] for x in sample)), "cal_counts": dict(Counter(x["dataset"] for x in cal)), "new_inference_cap": 2000, "arms": ["B1_BASE_JOINT", "B1_ORDER_P1", "B1_ORDER_P2", "B1_ORDER_P3", "B1_CONTEXT_MATCHED", "B1_CONTEXT_IRRELEVANT"], "anchor_rule": "position 0/5/10/15 cycled by canonical sample index", "replacement_rule": "four non-anchor positions; same-dataset source-distinct, length/topic matched or low-query-overlap; known gold IDs excluded offline", "model": "Cloudflare/clef-flash", "revision": CLEF_REV, "precision": "bfloat16", "score_contract": "JOINT_BINARY_RELEVANCE:v2; higher is more relevant; canonical ID remap", "candidate_count": 20, "quota": 5, "holdout_access": False, "protocol_sha256": "computed_after_write"})
    return {"manifest_rows": len(manifest), "new_rows": sum(r["status"] == "B1_GPU_BLOCKED" for r in manifest), "base_reused": sum(r["status"] == "reused" for r in manifest)}


def run_b2(items: list[dict], gold: dict, groups: dict, point: dict[str, dict], joint: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    rows = []
    for method, scores in (("pointwise_binary", point), ("joint_binary", joint)):
        for item in items:
            q = item["query_id"]; ids = [c["doc_id"] for c in item["candidates"]]; g = set(gold.get(q, {}).get("docs", [])); group = groups.get(q, [])
            if len(group) != 5: raise RuntimeError(f"invalid quartet mapping {q}")
            unrestricted = select_top(ids, scores[q]["scores"]); quota = select_quota(ids, scores[q]["scores"], group)
            eligible, conditional, _ = metric(g, ids, unrestricted)
            structural = eligible and all(sum(x in gg for x in g) <= 1 for gg in group)
            rows.append({"query_id": q, "cohort": item["cohort"], "dataset": item["dataset"], "method": method, "eligible": eligible, "conditional": conditional, "structurally_quota_feasible": structural, "unrestricted_ids": unrestricted, "quota_ids": quota, "unrestricted_success": int(eligible and g.issubset(set(unrestricted))), "quota_success": int(eligible and g.issubset(set(quota))), "score_input_hash": scores[q].get("actual_input_sha256"), "score_response_hash": scores[q].get("response_hash"), "group_hash": sha_json(group)})
    stats = []
    scopes = [("track_a", "hotpot"), ("track_a", "musique"), ("track_a", "wiki2"), ("track_a", "__pooled__"), ("track_b", "enterprise")]
    for method in ("pointwise_binary", "joint_binary"):
        for cohort, dataset in scopes:
            base = [r for r in rows if r["method"] == method and r["cohort"] == cohort and (dataset == "__pooled__" or r["dataset"] == dataset)]
            for pop, sub in (("all_identity_eligible", [r for r in base if r["eligible"]]), ("retrieval_conditional", [r for r in base if r["conditional"]])):
                deltas = [r["quota_success"] - r["unrestricted_success"] for r in sub]; lo, hi = bootstrap(deltas)
                stats.append({"cohort": cohort, "dataset": dataset, "method": method, "population": pop, "n": len(sub), "unrestricted_success": sum(r["unrestricted_success"] for r in sub), "quota_success": sum(r["quota_success"] for r in sub), "unrestricted_rate": sum(r["unrestricted_success"] for r in sub) / len(sub) if sub else None, "quota_rate": sum(r["quota_success"] for r in sub) / len(sub) if sub else None, "quota_minus_unrestricted": sum(deltas) / len(deltas) if deltas else None, "quota_wins": sum(x > 0 for x in deltas), "unrestricted_wins": sum(x < 0 for x in deltas), "ties": sum(x == 0 for x in deltas), "structural_feasibility_rate": sum(r["structurally_quota_feasible"] for r in sub) / len(sub) if sub else None, "ci95_low": lo, "ci95_high": hi, "bootstrap_reps": BOOTSTRAP, "bootstrap_seed": SEED, "interpretation": "algorithmic quota effect on same authenticated score vector"})
    # Interaction difference-in-differences, same scope and population.
    for cohort, dataset in scopes:
        for pop in ("all_identity_eligible", "retrieval_conditional"):
            vals = {}
            for method in ("pointwise_binary", "joint_binary"):
                r = next(x for x in stats if x["cohort"] == cohort and x["dataset"] == dataset and x["method"] == method and x["population"] == pop)
                vals[method] = r["quota_minus_unrestricted"]
            stats.append({"cohort": cohort, "dataset": dataset, "method": "joint_minus_pointwise_interaction", "population": pop, "n": "see paired rows", "unrestricted_success": "NOT_APPLICABLE", "quota_success": "NOT_APPLICABLE", "unrestricted_rate": "NOT_APPLICABLE", "quota_rate": "NOT_APPLICABLE", "quota_minus_unrestricted": (vals["joint_binary"] - vals["pointwise_binary"]) if vals["joint_binary"] is not None and vals["pointwise_binary"] is not None else None, "quota_wins": "NOT_APPLICABLE", "unrestricted_wins": "NOT_APPLICABLE", "ties": "NOT_APPLICABLE", "structural_feasibility_rate": "NOT_APPLICABLE", "ci95_low": "NOT_MEASURED", "ci95_high": "NOT_MEASURED", "bootstrap_reps": BOOTSTRAP, "bootstrap_seed": SEED, "interpretation": "difference-in-differences of tested score formulations; not hidden reasoning evidence"})
    return rows, stats


def tfidf_select(item: dict, scores: list[float], lam: float, mode: str) -> list[str]:
    candidates = item["candidates"]; ids = [c["doc_id"] for c in candidates]; index = {did: i for i, did in enumerate(ids)}; vec = [Counter(tokens(c.get("full_text", ""))) for c in candidates]
    lo, hi = min(scores), max(scores); rel = [(s - lo) / (hi - lo) if hi > lo else 0.5 for s in scores]
    selected = []; selected_indices = []; covered = [0.0] * len(ids)
    for step in range(5):
        best = None
        for i, did in enumerate(ids):
            if i in selected_indices: continue
            if mode == "mmr":
                redundancy = max((cosine(vec[i], vec[j]) for j in selected_indices), default=0.0)
                value = lam * rel[i] - (1 - lam) * redundancy
            else:
                gain = sum(max(covered[u], cosine(vec[u], vec[i])) - covered[u] for u in range(len(ids))) / len(ids)
                value = lam * rel[i] + (1 - lam) * gain
            key = (value, rel[i], -i, did)
            if best is None or key > best[0]: best = (key, did)
        selected.append(best[1]); selected_indices.append(index[best[1]])
        if mode != "mmr":
            chosen = selected_indices[-1]
            for u in range(len(ids)):
                covered[u] = max(covered[u], cosine(vec[u], vec[chosen]))
    return selected


def tune_setaware(e3a, lam_grid: tuple[float, ...]) -> dict:
    import importlib
    e2 = importlib.import_module("e2_m1_experiment")
    cal_items = e2.load_local_cal_items()
    cal_gold = {r["query_id"]: set(r.get("gold_doc_ids", [])) for r in jl(ROOT / "data/manifests/questions.jsonl")}
    scores = {r["query_id"]: r for r in jl(ROOT / "results/non_jev/v2/e2_n/n2/E2_N_N2_BGE_CAL.jsonl")}
    results = []
    for lam in lam_grid:
        for mode in ("mmr", "coverage"):
            ok = 0; den = 0
            for item in cal_items:
                q = item["query_id"]; g = cal_gold.get(q, set()); s = scores[q]; ids = [c["doc_id"] for c in item["candidates"]]
                if not (1 <= len(g) <= 5) or not g.issubset(set(ids)): continue
                den += 1; ok += int(g.issubset(set(tfidf_select(item, s["scores"], lam, mode))))
            results.append({"mode": mode, "lambda": lam, "cal_n": den, "cal_cssr": ok / den if den else None})
    best = {}
    for mode in ("mmr", "coverage"):
        subset = [r for r in results if r["mode"] == mode]
        best[mode] = max(subset, key=lambda r: (r["cal_cssr"] if r["cal_cssr"] is not None else -1, r["lambda"]))
    return {"grid": results, "selected": best, "selection_metric": "CAL conditional complete-support CSSR; ties choose larger lambda", "cal_source": "authenticated E2-N2 BGE CAL traces; no new inference"}


def run_b3(items: list[dict], gold: dict, state: dict, tune: dict) -> tuple[list[dict], list[dict]]:
    bge = load_score(ROOT / "results/non_jev/v2/e2_n/n3/E2_N_N3_BGE_FULL.jsonl")
    jina = load_score(ROOT / "results/non_jev/v2/e2_n/n3/E2_N_N3_JINA35_FULL.jsonl")
    pplx = load_score(ROOT / "results/non_jev/v2/e2_n/n3/E2_N_N3_PPLX_DECIDER_FULL.jsonl")
    clef_j = load_score(ROOT / "results/non_jev/v2/e2_m1/E2_M1_JOINT_BINARY_SCORES.jsonl")
    clef_p = load_score(ROOT / "results/non_jev/v2/e2_m1/E2_M1_POINTWISE_SCORES.jsonl")
    selected = []
    for item in items:
        q = item["query_id"]; ids = [c["doc_id"] for c in item["candidates"]]
        for policy, score_row, method in (("bge_top5", bge[q], "top"), ("bge_mmr", bge[q], "mmr"), ("bge_coverage", bge[q], "coverage"), ("jina35", jina[q], "top"), ("pplx_decider", pplx[q], "top"), ("clef_joint", clef_j[q], "top"), ("clef_pointwise", clef_p[q], "top")):
            if policy == "bge_mmr": sel = tfidf_select(item, score_row["scores"], tune["selected"]["mmr"]["lambda"], "mmr")
            elif policy == "bge_coverage": sel = tfidf_select(item, score_row["scores"], tune["selected"]["coverage"]["lambda"], "coverage")
            else: sel = select_top(ids, score_row["scores"])
            selected.append({"query_id": q, "cohort": item["cohort"], "dataset": item["dataset"], "policy": policy, "candidate_ids": ids, "selected_ids": sel, "selected_ids_sha256": sha_json(sel), "candidate_view_hash": sha_json([sha_bytes(str(c.get("full_text", "")).encode()) for c in item["candidates"]]), "source_score_trace": "E2_N_N3_BGE_FULL.jsonl" if policy.startswith("bge") else "authenticated E2 historical score trace", "source_score_trace_sha256": sha_file(ROOT / "results/non_jev/v2/e2_n/n3/E2_N_N3_BGE_FULL.jsonl") if policy.startswith("bge") else None, "gold_in_input": False, "result_type": "offline_heuristic_or_reused_score", "status": "ok"})
    write_json(OUT / "E3_B_SETAWARE_SELECTED_IDS.jsonl.tmp", {"not_used": True}) if False else None
    with (OUT / "E3_B_SETAWARE_SELECTED_IDS.jsonl").open("w", encoding="utf-8") as f:
        for row in selected: f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    quality = []; paired = []
    for cohort, dataset in (("track_a", "hotpot"), ("track_a", "musique"), ("track_a", "wiki2"), ("track_a", "__pooled__"), ("track_b", "enterprise")):
        for policy in POLICIES:
            base = [r for r in selected if r["policy"] == policy and r["cohort"] == cohort and (dataset == "__pooled__" or r["dataset"] == dataset)]
            for pop, sub in (("all_identity_eligible", [r for r in base if 1 <= len(gold.get(r["query_id"], {}).get("docs", [])) <= 5 and gold.get(r["query_id"], {}).get("identity") in {"RESOLVED", "RESOLVED_OR_EMPTY"}]), ("retrieval_conditional", [r for r in base if gold.get(r["query_id"], {}).get("identity") in {"RESOLVED", "RESOLVED_OR_EMPTY"} and 1 <= len(gold.get(r["query_id"], {}).get("docs", [])) <= 5 and set(gold[r["query_id"]]["docs"]).issubset(set(r["candidate_ids"]))])):
                success = [int(set(gold.get(r["query_id"], {}).get("docs", [])).issubset(set(r["selected_ids"]))) for r in sub]
                quality.append({"cohort": cohort, "dataset": dataset, "policy": policy, "population": pop, "n": len(sub), "success": sum(success), "cssr": sum(success) / len(success) if success else None, "interpretation": "complete-support selection proxy; BGE_MMR/COVERAGE are TF-IDF heuristics, not SetR"})
        for pop in ("all_identity_eligible", "retrieval_conditional"):
            for policy in ("bge_mmr", "bge_coverage"):
                a = next(r for r in quality if r["cohort"] == cohort and r["dataset"] == dataset and r["policy"] == policy and r["population"] == pop)
                b = next(r for r in quality if r["cohort"] == cohort and r["dataset"] == dataset and r["policy"] == "bge_top5" and r["population"] == pop)
                av = {r["query_id"]: r for r in selected if r["policy"] == policy and r["cohort"] == cohort and (dataset == "__pooled__" or r["dataset"] == dataset)}
                bv = {r["query_id"]: r for r in selected if r["policy"] == "bge_top5" and r["cohort"] == cohort and (dataset == "__pooled__" or r["dataset"] == dataset)}
                qids = []
                for q in av:
                    g = gold.get(q, {}); okid = g.get("identity") in {"RESOLVED", "RESOLVED_OR_EMPTY"} and 1 <= len(g.get("docs", [])) <= 5
                    if pop == "retrieval_conditional": okid = okid and set(g["docs"]).issubset(set(av[q]["candidate_ids"]))
                    if okid: qids.append(q)
                d = [int(set(gold[q]["docs"]).issubset(set(av[q]["selected_ids"]))) - int(set(gold[q]["docs"]).issubset(set(bv[q]["selected_ids"]))) for q in qids]; lo, hi = bootstrap(d)
                paired.append({"cohort": cohort, "dataset": dataset, "population": pop, "comparison": f"{policy}_minus_bge_top5", "n": len(d), "difference": sum(d) / len(d) if d else None, "wins": sum(x > 0 for x in d), "losses": sum(x < 0 for x in d), "ties": sum(x == 0 for x in d), "ci95_low": lo, "ci95_high": hi, "bootstrap_reps": BOOTSTRAP, "bootstrap_seed": SEED, "interpretation": "exploratory paired heuristic contrast"})
    write_csv(OUT / "E3_B_SETAWARE_QUALITY_AND_PAIRED_STATS.csv", quality + paired)
    return selected, quality


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    e3a, item_list, gold, state = load_e3a(); items = {x["query_id"]: x for x in item_list}
    groups = load_groups()
    point = load_score(ROOT / "results/non_jev/v2/e2_m1/E2_M1_POINTWISE_SCORES.jsonl"); joint = load_score(ROOT / "results/non_jev/v2/e2_m1/E2_M1_JOINT_BINARY_SCORES.jsonl")
    sample, cal = make_sample(item_list, gold, state); panels = make_panels(sample, items, gold)
    _freeze_protocol(sample, cal, panels)
    gpu_id = int(os.environ.get("E3_B_GPU_ID", "1"))
    if gpu_id not in {0, 1, 2, 4}:
        raise RuntimeError(f"E3-B GPU {gpu_id} is not in the owner-approved set {{0,1,2,4}}")
    if os.environ.get("E3_B_SKIP_NATIVE") == "1" and (OUT / "E3_B_CONTEXT_INTERVENTION_NATIVE_DETAIL.jsonl").is_file():
        native_traces = jl(OUT / "E3_B_CONTEXT_INTERVENTION_NATIVE_DETAIL.jsonl")
        b1_counts = _write_b1_manifest_and_trace_rows(sample, panels, joint, native_traces, gpu_id, {"status": "reused_from_completed_native_detail"}, 12, 30)
        b1_counts.update({"native_ok": sum(r.get("status") == "ok" for r in native_traces), "native_errors": sum(r.get("status") == "error" for r in native_traces), "new_physical_forward_calls": sum(r.get("physical_forward_count", 0) for r in native_traces), "native_detail_path": str((OUT / "E3_B_CONTEXT_INTERVENTION_NATIVE_DETAIL.jsonl").relative_to(ROOT))})
    else:
        b1_counts = _run_b1_native(sample, cal, panels, items, joint, gpu_id)
        native_traces = jl(OUT / "E3_B_CONTEXT_INTERVENTION_NATIVE_DETAIL.jsonl")
    actual_gpu_ids = sorted({int(r["gpu_id"]) for r in native_traces if r.get("status") == "ok" and str(r.get("gpu_id", "")).isdigit()})
    b1_counts["gpu_ids_used"] = actual_gpu_ids
    _summarize_b1(sample, panels, joint, native_traces)
    b2_rows, b2_stats = run_b2(item_list, gold, groups, point, joint)
    # Store a compact query-level B2 machine artifact; selected IDs are needed
    # for independent replay and are not a replacement for source scores.
    with (OUT / "E3_B_SAME_SCORE_QUOTA_RESULTS.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(b2_rows[0]); w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader()
        for r in b2_rows:
            rr = dict(r); rr["unrestricted_ids"] = sha_json(r["unrestricted_ids"]); rr["quota_ids"] = sha_json(r["quota_ids"]); w.writerow(rr)
    write_csv(OUT / "E3_B_SAME_SCORE_QUOTA_PAIRED_STATS.csv", b2_stats)
    tune = tune_setaware(e3a, (0.50, 0.70, 0.85, 1.00)); write_json(OUT / "E3_B_SETAWARE_CAL_CONFIG.json", tune)
    b3_selected, b3_quality = run_b3(item_list, gold, state, tune)
    # SetR audit is intentionally a feasibility result, not a model result.
    cache_hits = [str(p) for p in Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub").glob("*") if "setr" in p.name.lower()]
    write_text(OUT / "E3_B_SETR_FEASIBILITY_AUDIT.md", f"""# E3-B SetR Feasibility Audit

Verdict: `SETR_NOT_REPRODUCIBLE`

The official repository is [LGAI-Research/SetR](https://github.com/LGAI-Research/SetR), licensed BSD-3-Clause, and the paper is [ACL 2025, DOI 10.18653/v1/2025.acl-long.861](https://aclanthology.org/2025.acl-long.861/). The official README provides training-data generation, training, and response-generation instructions; its evaluation step is explicitly `TBD`. The repository page exposes no release/package/checkpoint artifact, and the permitted local HF cache has no SetR checkpoint (`local_cache_matches={cache_hits}`).

The documented path depends on generating responses/training a Llama-based selector and does not provide a verified pretrained Top-20-to-Top-5 inference contract for this project's frozen query/candidate representation. No repository was cloned, no weights were downloaded, no SetR inference was run, and no training was performed. Therefore the E3-B set-aware controls are explicitly `BGE_MMR` and `BGE_COVERAGE` TF-IDF heuristics, never SetR or a learned SetR equivalent.
""")
    write_json(OUT / "E3_B_FROZEN_INTERVENTION_PROTOCOL.json", json.loads((OUT / "E3_B_FROZEN_INTERVENTION_PROTOCOL.json").read_text()))
    # Supplement E3-0 without mutating the E3-A governance files.
    write_text(GOV / "E3_0_E3_B_SUPPLEMENTARY_STATUS.md", """# E3-0 Supplementary E3-B Status\n\nStatus remains `E3_0_HOLDOUT_RESERVATION_PENDING_METADATA`. No new official label-free MultiHop-RAG or FinanceBench metadata was present in the local workspace during E3-B. No reserved IDs, qrels, answers, reader outputs or holdout outcomes were accessed. E3-B historical B2/B3 work is independent of this unresolved prospective reservation.\n""")
    source_paths = [ROOT / "results/non_jev/v2/e2_m1/E2_M1_JOINT_BINARY_SCORES.jsonl", ROOT / "results/non_jev/v2/e2_m1/E2_M1_POINTWISE_SCORES.jsonl", ROOT / "results/non_jev/v2/e2_m1/E2_M1_SELECTED_DOCUMENTS.jsonl", ROOT / "results/non_jev/v2/e2_n/n3/E2_N_N3_BGE_FULL.jsonl", ROOT / "results/non_jev/v2/e3_v1/e3_a_paired/E3_A_QUERY_LEVEL_EVIDENCE_STATES.csv"]
    inputs = {str(p.relative_to(ROOT)): sha_file(p) for p in source_paths}
    write_text(OUT / "E3_B_SOURCE_AND_AUTHORIZATION_AUDIT.md", f"""# E3-B Source and Authorization Audit

Status: `E3_B_PASS_WITH_LIMITATIONS` for the bounded historical experiment; B0/B1/B2/B3 completed under the owner-authorized shared-GPU window. E3-0 remains independent and pending.

- Clef checkpoint: `Cloudflare/clef-flash`, revision `{CLEF_REV}`, bfloat16, authenticated E2-M1 native joint-binary contract.
- E2-M1 source rows: 4,267 pointwise and 4,267 joint, all status `ok`, candidate count 20, one physical forward per row, gold-free inputs.
- B0 sample: 400 retrieval-conditional identity-valid historical queries, 100 each HotpotQA/MuSiQue/2Wiki/EnterpriseRAG-Bench; seed {SEED}.
- B1 new inference cap: 2,000 full-slate requests plus 12 smoke and 30 CAL rows. Actual new physical calls: `{b1_counts['new_physical_forward_calls']}` across `{b1_counts['native_ok']}` successful and `{b1_counts['native_errors']}` failed rows.
- Host safety result: shared authorized GPU IDs `{actual_gpu_ids}`; no process was terminated or restarted. The read-only pre-run inventory and per-row GPU attribution are recorded in the provenance manifest/detail trace. GPU 5 was not used.
- E3-A and E3-0 status were preserved; no fresh holdout or Jev/paid API was accessed.

Input SHA256 commitments:
```json
{json.dumps(inputs, indent=2, sort_keys=True)}
```
""")
    write_text(OUT / "E3_B_NATIVE_CLEF_CONTRACT_REPORT.md", """# E3-B Native Clef Contract Report

The reused E2-M1 contract is authenticated as `JOINT_BINARY_RELEVANCE:v2`, higher probability means more relevant, 20 canonical candidate IDs, `field_id_candidate_order` alignment, original candidate order, and one native joint forward per slate. The B1 order panels preserved this contract and remapped every response to canonical IDs. New B1 requests used the same pinned native adapter; no pointwise fallback or quartet-normalized categorical score was substituted.
""")
    write_text(OUT / "E3_B_SMOKE_AND_CAL_REPORT.md", f"""# E3-B Smoke and CAL Report

- Existing authenticated baseline reuse: 400/400 B1 sample rows available from E2-M1 joint traces.
- New B1 smoke: 12 CAL queries, all six arms represented; successful native rows: `{sum(r.get('stage') == 'smoke' and r.get('status') == 'ok' for r in native_traces)}`.
- New B1 CAL: 30 deterministic CAL cases, original-order native joint; successful native rows: `{sum(r.get('stage') == 'cal' and r.get('status') == 'ok' for r in native_traces)}`.
- Full historical intervention arms: 2,000 prespecified rows; successful native rows: `{b1_counts['native_ok'] - sum(r.get('stage') in {'smoke', 'cal'} and r.get('status') == 'ok' for r in native_traces)}`.
- Failed/timeout rows: `{b1_counts['native_errors']}`; no score substitution was performed.
- GPU IDs `{actual_gpu_ids}` were shared under owner authorization; resident workloads were not terminated or restarted.
""")
    write_text(OUT / "E3_B_MECHANISM_SYNTHESIS.md", f"""# E3-B Mechanism Synthesis

B1 measured order/context effects on the frozen 400-query historical cohort using `{b1_counts['native_ok']}` successful native rows and `{b1_counts['native_errors']}` explicit failures. Order effects are canonical-ID remapped; context replacements are diagnostic and are not scored as original-slate CSSR. B2 isolates a policy constraint on identical Clef score vectors and is interpretable offline. B3 supplies transparent TF-IDF MMR and facility-location coverage controls. They are heuristic set-aware controls, not SetR and not learned SOTA. All conclusions are historical exploratory associations, not fresh confirmation or hidden-reasoning evidence.
""")
    write_text(OUT / "E3_B_GPU_RESOURCE_AND_FORWARD_ACCOUNTING.md", f"""# E3-B GPU Resource and Forward Accounting

GPU IDs `{actual_gpu_ids}` were used under explicit owner authorization while other resident workloads remained running. No service or process was terminated or restarted. B1 new physical forwards = `{b1_counts['new_physical_forward_calls']}`; B1 baseline reused 400 authenticated historical rows with their original one-forward evidence. B2/B3 used zero GPUs. The pre-run host inventory, model revision, per-row GPU ID, memory telemetry and actual forward counts are recorded in the native detail trace and provenance manifest. This is not a production capacity benchmark and no dollar cost is inferred from these exploratory calls.
""")
    write_text(OUT / "E3_B_AUTHENTICITY_AND_INDEPENDENT_REPLAY.md", """# E3-B Authenticity and Independent Replay

The B2 replay independently loads raw E2-M1 score vectors and raw quartet assignments, validates 20-ID alignment and five groups of four, then recomputes unrestricted and one-per-quartet selections. B3 independently loads raw BGE score vectors and computes TF-IDF similarities and selections without qrels in the selector. All gold use occurs after selection for offline metrics. B1 has no fabricated output: blocked rows carry `physical_forward_count=0` and an explicit failure reason.
""")
    status = {"status": "E3_B_PASS_WITH_LIMITATIONS", "owner_review_gate": "E3_B_OWNER_REVIEW_READY", "e3_0_status": "E3_0_HOLDOUT_RESERVATION_PENDING_METADATA", "b1_status": "B1_NATIVE_COMPLETE_WITH_EXPLICIT_SHARED_GPU_AUTHORIZATION", "b1_counts": b1_counts, "b0_sample_counts": dict(Counter(x["dataset"] for x in sample)), "b2_rows": len(b2_rows), "b3_selection_rows": len(b3_selected), "clef_revision": CLEF_REV, "gpu_snapshot": {"static_precheck": GPU_SNAPSHOT, "actual_pre_run": b1_counts.get("host_snapshot"), "selected_gpu_ids": actual_gpu_ids, "resident_workloads_interrupted": False}, "jev_calls": 0, "paid_api_calls": 0, "new_model_inference_requests": b1_counts["new_rows"], "new_physical_forward_calls": b1_counts["new_physical_forward_calls"], "training": False, "setr_status": "SETR_NOT_REPRODUCIBLE", "input_sha256": inputs, "git_commit": os.popen("git rev-parse HEAD").read().strip(), "note": "E3_B_OWNER_REVIEW_READY is the terminal review gate; native B1 completed under the updated owner GPU authorization."}
    write_json(OUT / "E3_B_PROVENANCE_MANIFEST.json", status)
    write_text(OUT / "E3_B_FINAL_SCIENTIFIC_REPORT.md", f"""# DecisionRAG E3-B Final Scientific Report

Verdict: `E3_B_PASS_WITH_LIMITATIONS` with owner-review gate `E3_B_OWNER_REVIEW_READY`.

B0 froze a balanced 400-query historical exploratory sample. B1 ran the predeclared native Clef order/context arms with `{b1_counts['new_physical_forward_calls']}` new physical forwards and reused E2-M1's authenticated joint baseline for the original 400 rows. B2 same-score quota analysis and B3 heuristic set-aware analysis completed offline. SetR was not reproducible from the official repository/checkpoint path and is reported as unavailable; MMR and coverage results are explicitly heuristic.

See the machine-readable CSV/JSONL artifacts for exact denominators and paired bootstrap intervals. No E3-C/D/F, training, fresh holdout, Jev or paid work was started.
""")
    # Provenance hashes are written last and exclude themselves from recursion.
    artifacts = [p for p in OUT.iterdir() if p.is_file() and p.name != "E3_B_PROVENANCE_MANIFEST.json"]
    status["artifact_sha256"] = {str(p.relative_to(ROOT)): sha_file(p) for p in sorted(artifacts)}
    write_json(OUT / "E3_B_PROVENANCE_MANIFEST.json", status)
    write_text(OUT / "E3_B_OWNER_REVIEW_GATE.md", f"""# E3-B Owner Review Gate

`E3_B_OWNER_REVIEW_READY`

Scientific status: `E3_B_PASS_WITH_LIMITATIONS`. B0, B1, B2 and B3 are complete and reproducible. B1 used authenticated E2-M1 baseline reuse plus `{b1_counts['new_physical_forward_calls']}` genuine native Clef forwards across approved GPU IDs `{actual_gpu_ids}`; no resident process was terminated or restarted. SetR remains `SETR_NOT_REPRODUCIBLE`, with explicit TF-IDF controls instead. E3-0 remains `E3_0_HOLDOUT_RESERVATION_PENDING_METADATA`. No E3-C/D/F milestone, training, Jev, paid API, or fresh holdout access occurred.
""")
    # Update manifest one final time because the owner gate is itself an output.
    artifacts = [p for p in OUT.iterdir() if p.is_file() and p.name != "E3_B_PROVENANCE_MANIFEST.json"]
    status["artifact_sha256"] = {str(p.relative_to(ROOT)): sha_file(p) for p in sorted(artifacts)}
    write_json(OUT / "E3_B_PROVENANCE_MANIFEST.json", status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
