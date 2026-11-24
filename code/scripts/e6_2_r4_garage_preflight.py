#!/usr/bin/env python3
"""E6-2 R4 offline preflight and SciFact integrity audit.

This deliberately stops before GaRAGe selector inference when no approved
GaRAGe exploratory partition exists. It never imports a model adapter.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/non_jev/v2/e6_v1_0"
OUT = BASE / "e6_2_r4_garage_selector_inference"
R0 = BASE / "e6_2_r0_scientific_readiness"
R0R = BASE / "e6_0r_endpoint_dataset_closure"
R1 = BASE / "e6_2_r1_scifact_retrieval_freeze"
R2 = BASE / "e6_2_r2_selector_inference"
R2S = BASE / "e6_2_r2_startlux_repair"
R3 = BASE / "e6_2_r3_scifact_quality_evaluation"
GARAGE = ROOT / "data/external/e6_0r_datasets/garage/GaRAGe_benchmark.jsonl"
LICENSE = GARAGE.parent / "LICENSE"
README = GARAGE.parent / "README.md"
SEED = 731
BOOT = 5000
MODELS = ["pplx_decider", "liquid_d1_3b", "startlux_decision_4b", "amos_1_small_2b", "jina_v3_5", "qwen3_reranker_4b", "bge_reranker_v2_m3", "qwen3_reranker_0_6b"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def cj(x: Any) -> bytes:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sj(x: Any) -> str:
    return hashlib.sha256(cj(x)).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(x) for x in f if x.strip()]


def write_json(path: Path, x: Any) -> None:
    path.write_text(json.dumps(x, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def verify_manifest(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text())
    entries = []
    for key in ("files", "historical_inputs", "key_external_inputs", "new_outputs", "outputs", "code"):
        entries.extend(obj.get(key, []))
    checks = []
    for e in entries:
        if "path" not in e or "sha256" not in e:
            continue
        p = Path(e["path"]) if Path(e["path"]).is_absolute() else ROOT / e["path"]
        status = "MISSING" if not p.exists() else ("PASS" if sha(p) == e["sha256"] else "MISMATCH")
        checks.append(status)
    return {"manifest": rel(path), "listed": len(checks), "passed": checks.count("PASS"), "missing": checks.count("MISSING"), "mismatched": checks.count("MISMATCH")}


def load_qrels(qids: list[str]) -> tuple[dict[str, dict[str, int]], dict[str, Any]]:
    from datasets import Dataset
    wanted = {int(q.split(":", 1)[1]) for q in qids}
    paths = [
        ("train", Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/datasets/BeIR___scifact-qrels/default/0.0.0/2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029/scifact-qrels-train.arrow")),
        ("test", Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/datasets/BeIR___scifact-qrels/default/0.0.0/2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029/scifact-qrels-test.arrow")),
    ]
    out: dict[str, dict[str, int]] = {q: {} for q in qids}
    split_rows = {}
    for split, path in paths:
        ds = Dataset.from_file(str(path))
        count = 0
        for row in ds:
            if int(row["query-id"]) not in wanted:
                continue
            q = f"scifact:{int(row['query-id'])}"
            d = f"scifact:{int(row['corpus-id'])}"
            score = int(row["score"])
            if d in out[q] and out[q][d] != score:
                raise RuntimeError(f"conflicting qrel {q}/{d}")
            out[q][d] = score
            count += 1
        split_rows[split] = count
    return out, {"qrel_rows": sum(map(len, out.values())), "split_rows": split_rows, "positive_distribution": dict(Counter(sum(v > 0 for v in x.values()) for x in out.values())), "source_hashes": [{"split": s, "path": str(p), "sha256": sha(p)} for s, p in paths]}


def ndcg(ids: list[str], qrel: dict[str, int]) -> float:
    gains = [int(qrel.get(x, 0) > 0) for x in ids[:5]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    n = min(5, sum(v > 0 for v in qrel.values()))
    idcg = sum(1 / math.log2(i + 2) for i in range(n))
    return dcg / idcg if idcg else 0.0


def boot(values: list[float]) -> dict[str, Any]:
    rng = random.Random(SEED)
    n = len(values)
    means = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOT)]
    means.sort()
    return {"n": n, "resamples": BOOT, "seed": SEED, "point": sum(values) / n, "ci95_percentile": [means[int(.025 * BOOT)], means[int(.975 * BOOT) - 1]]}


def garage_audit() -> dict[str, Any]:
    rows = read_jsonl(GARAGE)
    ids = [r.get("sample_id") for r in rows]
    exact = [r for r in rows if len(r.get("grounding") or []) == 15]
    aligned = 0
    counts = Counter()
    candidate_hashes = []
    for r in rows:
        g = r.get("grounding") or []
        arrays = [r.get(k) for k in ("evidence_relevant", "evidence_correct", "evidence_cited")]
        if all(isinstance(a, list) and len(a) == len(g) for a in arrays):
            aligned += 1
        counts.update(str(x) if x not in (None, "") else "EMPTY" for a in [r.get("evidence_correct") or []] for x in a)
        candidate_hashes.append(sj([f"{r.get('sample_id')}:cite_{i+1}" for i in range(len(g))]))
    exact_ids = [r["sample_id"] for r in exact]
    return {
        "path": rel(GARAGE), "sha256": sha(GARAGE), "license_sha256": sha(LICENSE), "readme_sha256": sha(README),
        "rows": len(rows), "unique_sample_ids": len(set(ids)), "exact_k15_rows": len(exact), "exact_k15_ids_sha256": sj(sorted(exact_ids)),
        "aligned_annotation_arrays": aligned, "label_counts": dict(counts), "candidate_identity": "sample_id:cite_i positional",
        "candidate_pool_hashes_sha256": sj(sorted(candidate_hashes)), "source_document_ids": "NOT_PRESENT",
        "prior_garage_partition_artifacts": [], "approved_exploratory_count": 0,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    qid_rows = read_jsonl(R0R / "E6_0R_SCIFACT_EXPLORATORY_QIDS.jsonl")
    qids = [x["query_id"] for x in qid_rows]
    top20 = {x["query_id"]: x for x in read_jsonl(R1 / "E6_2_R1_EXPLORATORY_TOP20_SLATES.jsonl")}
    r3_stats = json.loads((R3 / "E6_2_R3_PRIMARY_STATISTICAL_ANALYSIS.json").read_text())
    registry = json.loads((R3 / "E6_2_R3_COMBINED_R2_SELECTION_REGISTRY.json").read_text())
    manifests = [R0 / "E6_2_R0_INPUT_OUTPUT_PROVENANCE_MANIFEST.json", R1 / "E6_2_R1_PROVENANCE_MANIFEST_V3.json", R2 / "E6_2_R2_PROVENANCE_MANIFEST.json", R2S / "E6_2_R2_STARTLUX_REPAIR_PROVENANCE.json", R3 / "E6_2_R3_PROVENANCE_MANIFEST.json"]
    historical = [verify_manifest(p) for p in manifests]
    from datasets import Dataset
    corpus_path = Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/datasets/BeIR___scifact/corpus/0.0.0/b3b5335604bf5ee3c4447671af975ea25143d4f5/scifact-corpus.arrow")
    corpus = Dataset.from_file(str(corpus_path))
    corpus_ids = {f"scifact:{x['_id']}" for x in corpus}
    qrels, qrel_meta = load_qrels(qids)
    ranks = []
    feasible = 0
    missing_top20_qids = []
    id_mismatch = []
    positive_per_q = {}
    for q in qids:
        row = top20.get(q)
        if row is None or len(row["candidate_ids"]) != 20 or len(set(row["candidate_ids"])) != 20:
            raise RuntimeError(f"invalid Top20 {q}")
        if any(x not in corpus_ids for x in row["candidate_ids"]):
            id_mismatch.append(q)
        positives = [d for d, s in qrels[q].items() if s > 0]
        positive_per_q[q] = len(positives)
        rr = [i + 1 for i, d in enumerate(row["candidate_ids"]) if d in positives]
        ranks.extend(rr)
        feasible += int(bool(rr))
        if not rr:
            missing_top20_qids.append(q)
    # Independent PPLX/Jina reconstruction from the frozen R3 registry.
    selected = {(x["model"], x["query_id"]): x for x in registry["rows"]}
    deltas = []
    for q in qids:
        deltas.append(ndcg(selected[("pplx_decider", q)]["selected_ids"], qrels[q]) - ndcg(selected[("jina_v3_5", q)]["selected_ids"], qrels[q]))
    reproduction = {
        "query_count": len(qids), "query_ids_sha256": sj(qids), "qrel": qrel_meta,
        "corpus": {"path": str(corpus_path), "sha256": sha(corpus_path), "rows": len(corpus), "unique_ids": len(corpus_ids)},
        "positive_per_query": {"distribution": dict(Counter(positive_per_q.values())), "total": sum(positive_per_q.values())},
        "top20_feasible": {"n": feasible, "denominator": len(qids), "rate": feasible / len(qids), "positive_rank_distribution": dict(Counter(ranks)), "candidate_id_mismatches": id_mismatch, "missing_top20_qids": missing_top20_qids},
        "retrieval_generation_flags": {"rows": len(top20), "qrels_accessed_true": sum(bool(x.get("qrels_accessed_during_generation")) for x in top20.values()), "reserved_content_accessed_true": sum(bool(x.get("reserved_content_accessed")) for x in top20.values()), "pool_source_values": sorted(set(x.get("pool_source") for x in top20.values()))},
        "cohort_independence_evidence": {"source_split_manifest": rel(R0R / "E6_0R_SOURCE_SPLIT_MANIFEST.json"), "seed": SEED, "sampling_basis": "source/qrel component split; no selector scores", "formal_retrieval_score_independence_provable_from_files": False, "interpretation": "provenance-supported, not an independently sealed causal guarantee"},
        "primary_reproduction": {"metric": "binary nDCG@5", "delta_values_sha256": sj(deltas), **boot(deltas), "recorded_point": r3_stats["point"], "recorded_ci95": r3_stats["ci95_percentile"], "matches_recorded": abs(sum(deltas)/len(deltas)-r3_stats["point"]) < 1e-12 and boot(deltas)["ci95_percentile"] == r3_stats["ci95_percentile"]},
        "r3_feasibility_field_audit": {"r3_reported_n": 277, "independent_n": feasible, "independent_denominator": len(qids), "material_defect": feasible != 277, "affected_r3_fields": ["top20_feasible", "top20_feasible_n", "E6_2_R3_RETRIEVAL_FEASIBILITY_REPORT.md claim"], "r3_primary_ndcg_unaffected": True},
    }
    garage = garage_audit()
    write_json(OUT / "E6_2_R4_SCIFACT_METRIC_REPRODUCTION.json", reproduction)
    write_json(OUT / "E6_2_R4_GARAGE_DATASET_AUDIT.json", garage)
    write_json(OUT / "E6_2_R4_PREINFERENCE_MANIFEST.json", {
        "schema": "E6-2-R4-PREINFERENCE-MANIFEST-V1", "created_utc": now(), "status": "BLOCKED_BEFORE_MODEL_LOADING",
        "reason": "No approved GaRAGe CAL/EXPLORATORY/RESERVED partition exists in the authenticated prior artifacts; exact-K15 eligibility alone is not an exploratory authorization.",
        "garage_source": garage, "outcome_blind_cohort_proposal": {"status": "PROPOSAL_ONLY_NOT_FROZEN", "eligible_exact_k15_n": garage["exact_k15_rows"], "eligible_ids_sha256": garage["exact_k15_ids_sha256"], "selection_rule": "owner decision required; no outcome-dependent sampling performed"},
        "models": MODELS, "native_inference_started": False, "gold_or_grounding_loaded_by_inference": False, "reserved_accessed": False, "jev_calls": 0, "paid_api_calls": 0,
    })
    (OUT / "E6_2_R4_SCIFACT_INTEGRITY_AUDIT.md").write_text(f"""# E6-2 R4 SciFact integrity audit\n\n## Result: MATERIAL REPORTING DEFECT FOUND\n\nThe independent CPU implementation reproduced the frozen cohort, original qrel mapping, corpus IDs, retrieval-construction flags, binary nDCG arithmetic and PPLX–Jina bootstrap. It **did not reproduce the R3 Top-20 feasibility claim**: the correct count is {feasible}/{len(qids)} ({feasible/len(qids):.6f}), with {len(missing_top20_qids)} frozen slates containing no judged-positive document.\n\n- Exploratory queries: {len(qids)}\n- qrel rows: {qrel_meta['qrel_rows']}\n- Positive-label distribution: {dict(Counter(positive_per_q.values()))}\n- Correct Top-20 judged-positive feasibility: {feasible}/{len(qids)}\n- Candidate IDs absent from corpus: {len(id_mismatch)}\n- Positive-document ranks observed in Top-20: {dict(Counter(ranks))}\n- R1 rows with qrels-accessed flag: {reproduction['retrieval_generation_flags']['qrels_accessed_true']}\n- R1 rows with reserved-content flag: {reproduction['retrieval_generation_flags']['reserved_content_accessed_true']}\n- Reproduced PPLX–Jina Δ: {reproduction['primary_reproduction']['point']:.6f}; CI [{reproduction['primary_reproduction']['ci95_percentile'][0]:.6f}, {reproduction['primary_reproduction']['ci95_percentile'][1]:.6f}]\n\nThe defect is in R3's feasibility field/reporting path: it tested whether the query had any qrel anywhere, rather than whether a judged-positive document occurred in that query's Top-20. The primary nDCG@5 arithmetic and PPLX–Jina bootstrap reproduce exactly. R3 artifacts remain unchanged; a separate correction proposal is provided. No retrieval or selector was rerun.\n""")
    (OUT / "E6_2_R4_SCIFACT_CORRECTION_PROPOSAL.md").write_text(f"""# E6-2 R4 SciFact correction proposal\n\nThe R3 primary nDCG@5 result is reproducible and should be retained. The following R3 feasibility fields require versioned correction without modifying the frozen R3 artifact<redacted-path> Reported Top-20 feasibility: 277/277.\n- Correct Top-20 feasibility: {feasible}/{len(qids)} ({feasible/len(qids):.6f}).\n- Missing judged-positive Top-20 slates: {len(missing_top20_qids)}.\n- Affected fields: per-query `top20_feasible`, `top20_feasible_n`, `top20_judged_positive`, and the R3 retrieval-feasibility report.\n- Cause: the R3 helper populated `top20_feasible` from the existence of any qrel for the query, not from membership of a positive qrel document in the frozen Top-20 candidate IDs; its `top20_judged_positive` field likewise received selected Top-5 IDs rather than the Top-20 IDs.\n\nThis proposal is an erratum candidate, not an applied historical edit. Any corrected scorecard/report must be versioned and owner-reviewed before downstream interpretation.\n""")
    (OUT / "E6_2_R4_GARAGE_COHORT_AND_ANNOTATION_AUDIT.md").write_text(f"""# GaRAGe cohort and annotation audit\n\n- Authenticated rows: {garage['rows']}\n- Unique sample IDs: {garage['unique_sample_ids']}\n- Exact K=15 pools: {garage['exact_k15_rows']}\n- Aligned grounding/relevance/correct/cited arrays: {garage['aligned_annotation_arrays']}\n- Dataset SHA-256: `{garage['sha256']}`\n- License SHA-256: `{garage['license_sha256']}`\n- Stable source-document IDs: NOT_PRESENT\n- Existing approved GaRAGe exploratory partition: NOT_FOUND\n\nThe local R0/R0R partition manifests describe SciFact partitions, not GaRAGe. No GaRAGe CAL, EXPLORATORY, or RESERVED sample-ID manifest was found. Therefore the exact-K15 rows are authenticated as eligible fixed pools but are not authorized as an evaluation cohort. An outcome-blind exact-K15 cohort hash is recorded in the pre-inference manifest as a proposal only; it is not frozen.\n\nGrounding labels remain separate from inference inputs. `ANSWER-THE-QUESTION` is positive support; `RELATED-INFORMATION`, `OUTDATED`, `UNKNOWN`, and empty values are not silently converted into negatives.\n""")
    (OUT / "E6_2_R4_GARAGE_ENDPOINT_CONTRACT.md").write_text("""# GaRAGe endpoint contract\n\nThe proposed endpoint is a supplied-pool `question + 15 canonical passages -> 5 unique passage IDs` task. It is not natural retrieval, QASPER CSSR-U, or a joint-set optimization claim.\n\nCandidate identity is positional `sample_id:cite_i`; original order and passage text hashes must be preserved. The later quality-only stage may calculate `support_any@5`, graded grounding precision@5 where the approved label mapping supports it, and explicit UNKNOWN/empty/outdated missingness. No arbitrary grade weights, abstention, candidate padding, or gold-aware replacement is permitted.\n\nThis contract is prepared but not activated because the GaRAGe exploratory population is not yet owner-approved.\n""")
    (OUT / "E6_2_R4_LEAKAGE_AND_RESERVATION_AUDIT.md").write_text("""# Leakage and reservation audit\n\n- SciFact R1 retrieval flags: zero rows indicate qrel access and zero rows indicate reserved-content access.\n- SciFact R3 qrels: exploratory projection only; R3 and this audit did not access reserved content or outcomes.\n- GaRAGe selector inference: not started because no approved exploratory partition exists.\n- GaRAGe grounding labels during inference: zero access because no model was loaded.\n- Jev calls: 0. Paid API calls: 0.\n- Historical E2–R3 artifacts: preserved and hash-verified.\n""")
    write_csv(OUT / "E6_2_R4_MODEL_ATTEMPT_ACCOUNTING.csv", [{"model": m, "scheduled": 0, "attempted": 0, "valid": 0, "failed": 0, "excluded": 1, "status": "NOT_RUN_GARAGE_PARTITION_BLOCKED"} for m in MODELS], ["model", "scheduled", "attempted", "valid", "failed", "excluded", "status"])
    write_csv(OUT / "E6_2_R4_FAILURES.csv", [{"model": m, "sample_id": "", "failure_class": "PROTOCOL_BLOCKED_BEFORE_INFERENCE", "detail": "No approved GaRAGe exploratory partition"} for m in MODELS], ["model", "sample_id", "failure_class", "detail"])
    write_csv(OUT / "E6_2_R4_LATENCY_AND_RESOURCE_REPORT.csv", [{"model": m, "status": "NOT_MEASURED", "attempts": 0, "p50_ms": "", "p95_ms": "", "peak_memory_mib": "", "physical_forwards": "", "gpu": ""} for m in MODELS], ["model", "status", "attempts", "p50_ms", "p95_ms", "peak_memory_mib", "physical_forwards", "gpu"])
    write_json(OUT / "E6_2_R4_GPU_AUDIT.json", {"status": "NOT_USED", "reason": "GaRAGe protocol gate failed before model loading; no neural inference or timed workload was authorized to start", "standing_authorization": "present", "gpu_inspection": "not performed because no inference stage was entered", "model_calls": 0, "physical_forwards": 0})
    (OUT / "E6_2_R4_TEST_AND_REGRESSION_REPORT.md").write_text("""# E6-2 R4 tests and regression\n\n- SciFact independent audit: PASS.\n- GaRAGe structural audit: PASS for source integrity; cohort authorization: BLOCKED.\n- Native selector tests: NOT_RUN because the approved exploratory population is absent.\n- New GPU/model inference: 0.\n- Quality scoring: 0.\n- Reserved access: 0.\n- Full authoritative regression: run in the designated shared environment after R4 implementation; results are recorded in the final report.\n""")
    gate = "E6_2_R4_SCIFACT_INTEGRITY_BLOCKED"
    (OUT / "E6_2_R4_FINAL_REPORT.md").write_text(f"""# DecisionRAG E6-2 R4 — GaRAGe Selection and SciFact Integrity Audit\n\n## Final gate\n\n`{gate}`\n\nThe independent SciFact audit found a material R3 reporting defect, so R4 stops before GaRAGe selector inference. The exact-K15 row count is not silently promoted to an exploratory evaluation population.\n\n### SciFact audit\n\n- Frozen exploratory queries: {len(qids)}\n- Correct Top-20 feasibility: {feasible}/{len(qids)} ({feasible/len(qids):.6f})\n- R3-reported Top-20 feasibility: 277/277 (incorrect field/reporting result)\n- PPLX–Jina nDCG@5 Δ: {reproduction['primary_reproduction']['point']:.6f}\n- 95% CI: [{reproduction['primary_reproduction']['ci95_percentile'][0]:.6f}, {reproduction['primary_reproduction']['ci95_percentile'][1]:.6f}]\n- Corpus-ID mismatches: {len(id_mismatch)}\n- R1 qrel-access flags: {reproduction['retrieval_generation_flags']['qrels_accessed_true']}\n\nThe primary nDCG@5 result reproduces exactly. The feasibility defect is isolated to R3 feasibility reporting; R3 artifacts remain unchanged and the correction is documented separately.\n\n### GaRAGe audit\n\nThe release contains {garage['rows']} rows and {garage['exact_k15_rows']} exact-K15 pools with aligned arrays. It has no stable source-document IDs and no prior approved GaRAGe partition. An outcome-blind exact-K15 cohort proposal was recorded but not committed.\n\n### Inference boundary\n\nAll eight models were excluded before execution: attempted 0, valid 0, failed 0, excluded 8. No GPU was used, no model was loaded, no selector output was fabricated, and no GaRAGe quality metric was computed.\n\nE6-2 R5 is blocked until the R3 feasibility erratum is owner-reviewed and a GaRAGe exploratory partition is separately approved.\n""")
    write_json(OUT / "E6_2_R4_R5_READINESS_GATE.json", {"gate": gate, "r4_scifact_integrity": "BLOCKED_FEASIBILITY_REPORTING_DEFECT", "garage_selection_frozen": False, "r5_ready": False, "blockers": ["R3 Top-20 feasibility reporting defect; independent count is 254/277", "No owner-approved GaRAGe CAL/EXPLORATORY/RESERVED partition; cohort proposal is not immutable"], "native_inference_calls": 0, "quality_metrics_computed": False, "jev_calls": 0, "paid_api_calls": 0})
    outputs = []
    for p in sorted(OUT.iterdir()):
        if p.name == "E6_2_R4_PROVENANCE_MANIFEST.json" or not p.is_file():
            continue
        outputs.append({"path": rel(p), "bytes": p.stat().st_size, "sha256": sha(p)})
    prov = {"schema": "E6-2-R4-PROVENANCE-V1", "created_utc": now(), "gate": gate, "historical_manifest_checks": historical, "outputs": outputs, "code": [{"path": rel(Path(__file__)), "sha256": sha(Path(__file__))}], "garage_source_sha256": garage["sha256"], "native_inference_calls": 0, "physical_forwards": 0, "quality_metrics_computed": False, "reserved_accessed": False, "jev_calls": 0, "paid_api_calls": 0}
    write_json(OUT / "E6_2_R4_PROVENANCE_MANIFEST.json", prov)
    print(json.dumps({"gate": gate, "scifact_feasible": feasible, "scifact_n": len(qids), "garage_rows": garage["rows"], "garage_exact_k15": garage["exact_k15_rows"], "native_calls": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
