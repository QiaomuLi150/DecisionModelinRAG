#!/usr/bin/env python3
"""CPU-only preparation for DecisionRAG v1.3 follow-up experiments.

This script freezes only label-blind inputs and protocols.  It deliberately
does not import model runtimes, load checkpoints, call services, or read
qrels/answers.  Inference stages must consume these append-only artifacts.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v1_3"
BRIGHT_PATH = ROOT / "results/non_jev/v2/e9_bright_v1_0/01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl"
E12_LOCK = ROOT / "results/non_jev/v2/e12_o_final_v1_0/00_protocol/E12O_COHORT_LOCK.json"
E11_COHORT = ROOT / "results/non_jev/v2/e11v_v1_0/01_admission/V1_COHORT_MANIFEST.json"
E11_CONTENT = ROOT / "results/non_jev/v2/e11v_v1_0/02_selection/V1_COHORT_CONTENT_VISIBLE_TEXT.jsonl"

PINNED = {
    "pplx_decider": {
        "revision": "14c9d90137962d300f3de95633f7b6535e80c5ce",
        "short_revision": "14c9d90",
    },
    "kev9": {"revision": "db029f08b290afd9fee4aa4bbcd9ae48602d1eb0", "short_revision": "db029f0"},
    "drex_dlm": {"revision": "8f6f8244ce0b186b3caa4cdbaae1c9f3986bf63e", "short_revision": "8f6f824"},
    "winnow12": {"revision": "0b5b7f98e718cf0faffd0acea3093b86ebf00b56", "short_revision": "0b5b7f9"},
    "jina_v3_5": {"revision": "e8a93f33f0b22108f8c2364f8484ce3422552fbc", "short_revision": "e8a93f3"},
    "qwen3_reranker_0_6b": {"revision": "e61197ed45024b0ed8a2d74b80b4d909f1255473", "short_revision": "e61197e"},
}


def canon(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_obj(value) -> str:
    return sha_bytes(canon(value).encode("utf-8"))


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_bright():
    return [json.loads(line) for line in BRIGHT_PATH.open(encoding="utf-8") if line.strip()]


def candidate_ids(row):
    return [str(c["candidate_id"]) for c in row["candidates"]]


def candidate_text_hash(row):
    return sha_obj([str(c.get("content", "")) for c in row["candidates"]])


def row_record(row):
    ids = candidate_ids(row)
    return {
        "canonical_id": f"{row['domain']}:{row['query_id']}",
        "query_id": str(row["query_id"]),
        "domain": row["domain"],
        "query_sha256": row.get("query_sha256"),
        "input_sha256": row.get("input_sha256"),
        "slate_sha256": row.get("slate_sha256"),
        "candidate_ids": ids,
        "candidate_ids_sha256": sha_obj(ids),
        "candidate_texts_sha256": candidate_text_hash(row),
        "candidate_order": "original_frozen_bm25_order",
        "labels_joined": False,
    }


def build_permutations(records):
    out = []
    for r in records:
        ids = list(r["candidate_ids"])
        orders = [("original", ids), ("reversed", list(reversed(ids)))]
        for seed in range(1, 6):
            p = list(ids)
            random.Random(seed).shuffle(p)
            orders.append((f"perm_seed_{seed}", p))
        for order_id, ordered in orders:
            out.append({
                "canonical_id": r["canonical_id"],
                "domain": r["domain"],
                "query_id": r["query_id"],
                "order_id": order_id,
                "seed": None if order_id in {"original", "reversed"} else int(order_id.rsplit("_", 1)[1]),
                "candidate_ids": ordered,
                "candidate_ids_sha256": sha_obj(ordered),
                "source_slate_sha256": r["slate_sha256"],
                "source_input_sha256": r["input_sha256"],
                "labels_joined": False,
            })
    return out


def build_two_stage(records):
    out = []
    for r in records:
        ids = r["candidate_ids"]
        windows = [ids[:10], ids[10:20]]
        out.append({
            "canonical_id": r["canonical_id"],
            "domain": r["domain"],
            "query_id": r["query_id"],
            "policy": "pplx_two_stage",
            "stage_windows": [
                {"stage": 1, "window": 1, "candidate_ids": windows[0], "candidate_ids_sha256": sha_obj(windows[0])},
                {"stage": 1, "window": 2, "candidate_ids": windows[1], "candidate_ids_sha256": sha_obj(windows[1])},
            ],
            "stage_2_candidate_count": 10,
            "budget_rule": {"L": 8192, "rho": 128, "question_shortening": False, "redistribution": False},
            "source_slate_sha256": r["slate_sha256"],
            "source_input_sha256": r["input_sha256"],
            "labels_joined": False,
        })
    return out


def protocol(experiment_id, description, population, analysis, deviations=None):
    return {
        "protocol_id": f"decisionrag-v1.3-{experiment_id}-v1.0",
        "experiment_id": experiment_id,
        "status": "FROZEN_PRE_OUTCOME",
        "description": description,
        "population": population,
        "pinned_models": PINNED,
        "selection_validity": "exactly five distinct original candidate IDs; every failure retained; no silent retry/fallback",
        "label_blind": True,
        "budget_rule": {"L": 8192, "rho": 128, "K": "number_of_candidates", "b0_is_failure": True, "backoff": "one_token_at_a_time"},
        "timing_rule": "dedicated A100-SXM4-80GB, no resident process before/after, one request at a time, exclude model load from latency but report it",
        "determinism_rule": "20 randomly selected non-timing requests rerun after raw-output freeze",
        "analysis": analysis,
        "deviations": deviations or [],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_bright()
    records = [row_record(row) for row in rows]
    assert len(records) == 1384, len(records)
    assert len({r["canonical_id"] for r in records}) == len(records)
    assert all(len(r["candidate_ids"]) == 20 and len(set(r["candidate_ids"])) == 20 for r in records)

    lock = json.loads(E12_LOCK.read_text(encoding="utf-8"))
    serving_ids = list(lock["bright"]["serving_ids"])
    serving_records = [r for r in records if r["canonical_id"] in set(serving_ids)]
    assert len(serving_records) == 120, len(serving_records)

    cohort = json.loads(E11_COHORT.read_text(encoding="utf-8"))
    crag_counts = {k: len(v) for k, v in cohort.items() if isinstance(v, list)}

    source_inventory = {
        "schema": "decisionrag-v1.3-source-inventory-v1",
        "created_by": "scripts/v1_3_prepare.py",
        "label_blind": True,
        "inputs": {
            "bright_frozen_top20_slates": {"path": str(BRIGHT_PATH.relative_to(ROOT)), "sha256": file_sha(BRIGHT_PATH), "records": len(records)},
            "e12o_cohort_lock": {"path": str(E12_LOCK.relative_to(ROOT)), "sha256": file_sha(E12_LOCK), "serving_ids": len(serving_ids)},
            "e11v_cohort_manifest": {"path": str(E11_COHORT.relative_to(ROOT)), "sha256": file_sha(E11_COHORT), "counts": crag_counts},
        },
        "bright_domains": dict(sorted(Counter(r["domain"] for r in records).items())),
        "pinned_models": PINNED,
        "protected_inputs_not_read": ["qrels", "gold answers", "reference answers", "prior outcome labels"],
    }
    write_json(OUT / "preflight/source_inventory.json", source_inventory)
    write_json(OUT / "preflight/bright_frozen_manifest.json", {"records": records, "count": len(records), "sha256": sha_obj(records), "labels_joined": False})
    write_json(OUT / "preflight/e12o_serving_manifest.json", {"records": serving_records, "count": len(serving_records), "source_cohort_sha256": file_sha(E12_LOCK), "labels_joined": False})

    r1 = protocol("r1_serving_gate", "E12-O 120-query serving cohort, three attempts, matched selector timing", {"canonical_ids": [r["canonical_id"] for r in serving_records], "attempts": 3, "concurrency": 1}, {"primary": "pplx_budget versus jina_full p95 and p50 latency", "secondary": "jina_pplx_budget", "marginal_cost": True, "export_columns": ["session", "policy", "attempts", "valid", "p50_ms", "p95_ms", "fail_p50_ms", "alloc_p95_mib", "load_ms", "marginal_p50_ms", "marginal_alloc_mib"]})
    r2 = protocol("r2_order_sensitivity", "BRIGHT original, reverse and five seeded candidate-order permutations", {"records": len(records), "orders_per_query": 7, "seeds": [1, 2, 3, 4, 5], "permutation_manifest": "r2_order_sensitivity/permutation_manifest.jsonl"}, {"primary": "order robustness and known-positive nDCG@5", "aggregation": "Borda across six permutations plus original", "intervals": "40,000 whole-domain bootstrap over 12 domains plus exact 2^12 sign flip", "holm_family": "v1.3-order-contrasts"})
    r3 = protocol("r3_multipass", "PPLX two-stage 20-to-10-plus-10 then 10-to-5", {"records": len(records), "calls_per_query": "3 on success; any failed call fails request"}, {"primary": "known-positive nDCG@5", "timing": "R1 cohort under R1 dedicated-GPU protocol"})
    r4 = protocol("r4_crag_reader_window", "E11-V frozen CRAG selections with fixed Qwen reader at 256, 512 and 1024 tokens/page", {"representative": crag_counts.get("representative", 0), "oversized_diagnostic": crag_counts.get("oversized_diagnostic", 0), "existing_reference_window": 256, "new_windows": [512, 1024]}, {"reader_prompt_revision": "E11V_FIXED_READER_QWEN_V1", "decoding": "greedy", "max_new_tokens": 64, "intervals": "5,000 source-group bootstrap, seed 731"}, ["No new reader inference is authorized until a dedicated GPU timing/inference gate is available."])
    r5 = protocol("r5_winnow_budget", "Winnow-12B E11-A budget and Jina-on-identical-strings controls", {"cohort": "BRIGHT frozen Top-20", "server_build": "must be authenticated from E12 native manifest before execution"}, {"primary": "known-positive nDCG@5", "diagnostics": "HTTP error class plus classified server stderr", "binary_hash_required": True})
    r6 = protocol("r6_leetcode_trim", "Label-blind longest native-fit prefix for 142 LeetCode queries", {"domain": "leetcode", "expected_records": 142, "minimum_subslate_size": 6}, {"primary": "within-sub-slate known-positive nDCG@5", "contrasts": ["model-minus-Jina untrimmed", "model-minus-Jina trimmed", "trimmed-minus-untrimmed"]}, ["Native Kev/Drex tokenizer feasibility must be established before immutable sub-slate commitment."])
    for p in (r1, r2, r3, r4, r5, r6):
        write_json(OUT / p["experiment_id"] / "PROTOCOL.json", p)

    permutation_rows = build_permutations(records)
    write_jsonl(OUT / "r2_order_sensitivity/permutation_manifest.jsonl", permutation_rows)
    write_json(OUT / "r2_order_sensitivity/permutation_manifest.lock.json", {"count": len(permutation_rows), "sha256": file_sha(OUT / "r2_order_sensitivity/permutation_manifest.jsonl"), "labels_joined": False})
    write_jsonl(OUT / "r3_multipass/multipass_manifest.jsonl", build_two_stage(records))
    write_json(OUT / "r3_multipass/multipass_manifest.lock.json", {"count": len(records), "sha256": file_sha(OUT / "r3_multipass/multipass_manifest.jsonl"), "labels_joined": False})

    leetcode = [r for r in records if r["domain"] == "leetcode"]
    write_json(OUT / "r6_leetcode_trim/leetcode_candidates_preparation.json", {"records": leetcode, "count": len(leetcode), "status": "BLOCKED_BEFORE_TOKENIZER_FREEZE", "reason": "No model inference or native-tokenizer execution has been performed in this CPU-only preparation stage; a native Kev/Drex tokenizer feasibility run is required before freezing sub-slates.", "labels_joined": False})

    readme = {
        "status": "PARTIAL_PREPARATION_BLOCKED_FOR_INFERENCE",
        "created_by": "scripts/v1_3_prepare.py",
        "frozen_inputs": {"bright": len(records), "serving": len(serving_records), "crag": crag_counts, "leetcode": len(leetcode)},
        "prepared": ["R1-R6 protocol locks", "BRIGHT label-blind manifest", "E12-O serving manifest", "R2 permutation manifest", "R3 two-stage manifest", "R6 candidate preparation"],
        "inference_status": {"r1": "BLOCKED_DEDICATED_GPU_NOT_AVAILABLE", "r2": "BLOCKED_SAFE_NATIVE_EXECUTION_WINDOW_NOT_AVAILABLE", "r3": "BLOCKED_SAFE_NATIVE_EXECUTION_WINDOW_NOT_AVAILABLE", "r4": "BLOCKED_DEDICATED_GPU_NOT_AVAILABLE", "r5": "BLOCKED_SERVER_AND_GPU_GATE", "r6": "BLOCKED_NATIVE_TOKENIZER_FREEZE"},
        "protection": "No qrels, answers, reference labels or prior outcomes were read; no model runtime was imported; no model/service request was made.",
        "source_sha256": source_inventory["inputs"],
    }
    write_json(OUT / "README.json", readme)
    (OUT / "README.md").write_text("""# DecisionRAG v1.3 follow-up experiments\n\n## Current gate\n\n`PARTIAL_PREPARATION_BLOCKED_FOR_INFERENCE`\n\nThe append-only protocols and label-blind manifests are frozen. No model inference, model-service request, qrel/answer read, or paid API call was made by this preparation run. The timing gate requires a dedicated A100-SXM4-80GB with no resident process; the current hardware snapshot shows resident unrelated processes on every GPU.\n\nPrepared artifacts are listed in `README.json`. R1 and R4 require a dedicated GPU. R2/R3/R5/R6 require safe native execution and, for R6, a native tokenizer feasibility result before sub-slate commitment. No result CSV is created until actual inference outputs exist.\n""", encoding="utf-8")

    print(json.dumps({"status": readme["status"], "bright": len(records), "serving": len(serving_records), "permutation_rows": len(permutation_rows), "leetcode": len(leetcode), "source_sha256": file_sha(BRIGHT_PATH)}, indent=2))


if __name__ == "__main__":
    main()
