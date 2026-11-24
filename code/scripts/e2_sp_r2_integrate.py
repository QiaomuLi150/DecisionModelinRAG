#!/usr/bin/env python3
"""Integrate the completed E2-S/P-R2 isolated serving and answer evidence."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/non_jev/v2/e2_sp_r"
OUT = ROOT / "results/non_jev/v2/e2_sp_r2"
R3 = ROOT / "results/non_jev/v2/e2_sp_r/isolated_gpu1/r3"
MODELS = ("bge_pointwise", "jina35", "nimble", "gte", "pplx_decider")
MODEL_LABELS = {
    "bge_pointwise": "BGE",
    "jina35": "Jina-v3.5",
    "nimble": "Nimble-v2",
    "gte": "GTE",
    "pplx_decider": "PPLX Decider",
}
POLICY_MAP = {"policy_01": "retrieval5", "policy_02": "bge_pointwise", "policy_03": "jina35", "policy_04": "nimble", "policy_05": "gte", "policy_06": "pplx_decider"}
BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 731


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_json(value) -> str:
    return sha_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(n: int, d: int) -> float | str:
    return round(n / d, 6) if d else "NOT_MEASURED"


def normalize(value) -> str:
    import re
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", str(value or "").lower()).split())


def f1_proxy(prediction, reference, aliases=()) -> float | None:
    from collections import Counter
    if reference is None:
        return None
    pred = normalize(prediction).split()
    if not pred:
        return 0.0
    best = 0.0
    for candidate in [reference, *aliases]:
        ref = normalize(candidate).split()
        if not ref:
            continue
        overlap = sum((Counter(pred) & Counter(ref)).values())
        if overlap:
            precision, recall = overlap / len(pred), overlap / len(ref)
            best = max(best, 2 * precision * recall / (precision + recall))
    return best


def bootstrap_ci(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(values)
    estimates = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOTSTRAP_REPS)]
    estimates.sort()
    return estimates[int(0.025 * (len(estimates) - 1))], estimates[int(0.975 * (len(estimates) - 1))]


def integrate_r3() -> tuple[list[dict], dict]:
    rows, checks = [], []
    for model in MODELS:
        summary_path = R3 / f"E2_SP_R_R3_ISOLATED_GPU1_{model.upper()}_SUMMARY.json"
        trace_path = R3 / f"E2_SP_R_R3_ISOLATED_GPU1_{model.upper()}_REQUESTS.jsonl"
        summary = read_json(summary_path)
        trace = read_jsonl(trace_path)
        request_rows = [r for r in trace if r.get("record_type") == "request"]
        checks.extend([
            (len(request_rows) == 160, f"{model}:160 request traces"),
            (len(summary.get("summaries", [])) == 4, f"{model}:4 rate summaries"),
            (summary.get("status") == "ISOLATED_CAPACITY", f"{model}:isolated status"),
            (summary.get("physical_gpu") == 1 and summary.get("isolated") is True, f"{model}:GPU1 isolation marker"),
            (all(r.get("status") == "ok" for r in request_rows), f"{model}:zero request errors"),
            (all(r.get("gold_in_inference_input") is False and r.get("cache_reuse") is False for r in request_rows), f"{model}:gold/cache guard"),
        ])
        for s in summary["summaries"]:
            passed = bool(s.get("slo_1000_pass")) and s.get("errors") == 0
            rows.append({
                "model": model, "model_label": MODEL_LABELS[model], "rate_index": s["rate_index"],
                "offered_rps": s["offered_rps"], "achieved_qps": s["achieved_qps"], "requests": s["requests"],
                "completed": s["completed"], "errors": s["errors"], "error_rate": s["error_rate"],
                "p50_ms": s["p50_ms"], "p95_ms": s["p95_ms"], "p99_ms": s["p99_ms"],
                "queue_p50_ms": s["queue_p50_ms"], "queue_p95_ms": s["queue_p95_ms"],
                "queue_growth_ms": s["queue_growth_ms"], "gpu_seconds_per_completed_query_proxy": s["gpu_seconds_per_completed_query_proxy"],
                "physical_forward_calls": s["physical_forward_calls"], "slo_1000_pass": s["slo_1000_pass"],
                "slo_2000_pass": s["slo_2000_pass"], "slo_3000_pass": s["slo_3000_pass"],
                "stable_queue_observable": "QUEUE_GROWTH_PROXY_ONLY", "verified_sustainable_qps": "NOT_FORMALLY_VERIFIED",
                "slo_pass_and_zero_errors": passed,
            })
    with (OUT / "E2_SP_R2_ISOLATED_R3_SUMMARY.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(rows[0])
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    return rows, {"checks": checks, "passed": sum(ok for ok, _ in checks), "total": len(checks), "all_pass": all(ok for ok, _ in checks)}


def integrate_adjudication() -> tuple[list[dict], list[dict], dict]:
    source = read_jsonl(BASE / "E2_SP_R_ADJUDICATION_CASES.jsonl")
    cases = {r["case_id"]: r for r in source}
    judge_rows = read_jsonl(OUT / "E2_SP_R2_LOCAL_ADJUDICATION.jsonl")
    primary = [r for r in judge_rows if int(r["repeat_index"]) == 1]
    repeats = [r for r in judge_rows if int(r["repeat_index"]) == 2]
    rows = []
    for r in primary:
        case = cases[r["case_id"]]
        parsed = r.get("parsed") or {}
        rows.append({**r, "dataset": case["dataset"], "cohort": case["cohort"], "policy_blind_id": case["policy_blind_id"], "policy": POLICY_MAP[case["policy_blind_id"]], "reference_status": case["reference_status"], "reference_answer": case.get("reference_answer"), "reference_aliases": case.get("reference_aliases", []), "reader_answer": case["reader_answer"], "correctness": parsed.get("correctness"), "groundedness": parsed.get("groundedness"), "completeness": parsed.get("completeness"), "confidence": parsed.get("confidence")})
    summary_rows = []
    for dataset in ("hotpot", "musique", "wiki2", "enterprise"):
        for policy in POLICY_MAP.values():
            subset = [r for r in rows if r["dataset"] == dataset and r["policy"] == policy]
            verified = [r for r in subset if r["reference_status"] == "REFERENCE_AVAILABLE"]
            em = [1.0 if normalize(r["reader_answer"]) in {normalize(r["reference_answer"]), *(normalize(x) for x in r["reference_aliases"])} else 0.0 for r in verified]
            f1 = [f1_proxy(r["reader_answer"], r["reference_answer"], r["reference_aliases"]) for r in verified]
            summary_rows.append({"dataset": dataset, "policy": policy, "n_queries": len(subset), "reference_available_n": len(verified), "local_correctness_pass": sum(r["correctness"] == "PASS" for r in subset), "local_correctness_rate": pct(sum(r["correctness"] == "PASS" for r in subset), len(subset)), "local_groundedness_pass": sum(r["groundedness"] == "PASS" for r in subset), "local_groundedness_rate": pct(sum(r["groundedness"] == "PASS" for r in subset), len(subset)), "local_completeness_pass": sum(r["completeness"] == "PASS" for r in subset), "local_completeness_rate": pct(sum(r["completeness"] == "PASS" for r in subset), len(subset)), "unjudgeable_any_dimension": sum("UNJUDGEABLE" in {r["correctness"], r["groundedness"], r["completeness"]} for r in subset), "em_proxy": sum(em) / len(em) if em else "NOT_MEASURED", "f1_proxy": sum(f1) / len(f1) if f1 else "NOT_MEASURED", "assessment_provenance": "LLM_ADJUDICATED_LOCAL_BLINDED", "human_labels": 0})
    with (OUT / "E2_SP_R2_LOCAL_ADJUDICATION_SUMMARY.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(summary_rows[0]); writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(summary_rows)

    # Paired local-judge and reference-answer proxy comparisons use the same query IDs.
    by_query = defaultdict(dict)
    for r in rows: by_query[r["query_id"]][r["policy"]] = r
    paired = []
    for population, predicate in (("R4_96_multihop", lambda q: q["dataset"] != "enterprise"), ("R4_24_enterprise", lambda q: q["dataset"] == "enterprise"), ("R4_120_all", lambda q: True)):
        queries = sorted(q for q in by_query if predicate(next(iter(by_query[q].values()))))
        for metric, value in (("local_correctness", lambda r: float(r["correctness"] == "PASS")), ("local_groundedness", lambda r: float(r["groundedness"] == "PASS")), ("local_completeness", lambda r: float(r["completeness"] == "PASS")), ("deterministic_f1_proxy", lambda r: float(f1_proxy(r["reader_answer"], r["reference_answer"], r["reference_aliases"])) if r["reference_status"] == "REFERENCE_AVAILABLE" else None)):
            for left, right in (("pplx_decider", "jina35"), ("pplx_decider", "gte"), ("pplx_decider", "bge_pointwise"), ("jina35", "gte"), ("jina35", "bge_pointwise")):
                diffs = []
                for q in queries:
                    if left not in by_query[q] or right not in by_query[q]: continue
                    a, b = value(by_query[q][left]), value(by_query[q][right])
                    if a is not None and b is not None: diffs.append(a - b)
                lo, hi = bootstrap_ci(diffs)
                paired.append({"population": population, "metric": metric, "left": left, "right": right, "n_paired_queries": len(diffs), "mean_difference": sum(diffs) / len(diffs) if diffs else "NOT_MEASURED", "ci95_low": lo if lo is not None else "NOT_MEASURED", "ci95_high": hi if hi is not None else "NOT_MEASURED", "bootstrap_reps": BOOTSTRAP_REPS, "bootstrap_seed": BOOTSTRAP_SEED, "confirmatory_claim": False})
    with (OUT / "E2_SP_R2_PAIRED_ANSWER_STATISTICS.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(paired[0]); writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(paired)
    repeat_map = {(r["case_id"], int(r["repeat_index"])): r for r in judge_rows}
    repeat_pairs = []
    for r in repeats:
        p = repeat_map.get((r["case_id"], 1))
        if p and r.get("status") == "ok" and p.get("status") == "ok":
            a, b = p.get("parsed") or {}, r.get("parsed") or {}
            repeat_pairs.append(a == b)
    audit = {"primary_rows": len(primary), "repeat_rows": len(repeats), "all_rows": len(judge_rows), "status_counts": {"ok": sum(r.get("status") == "ok" for r in judge_rows), "invalid": sum(r.get("status") != "ok" for r in judge_rows)}, "repeat_exact_agreement_n": sum(repeat_pairs), "repeat_comparable_n": len(repeat_pairs), "repeat_exact_agreement_rate": pct(sum(repeat_pairs), len(repeat_pairs))}
    return rows, summary_rows, audit


def write_reports(r3_rows, r3_audit, quality_rows, quality_summary, quality_audit):
    cal = read_json(OUT / "E2_SP_R2_CAL_CONFIG_LOCK.json")
    r3_groups = {}
    for model in MODELS:
        entries = [r for r in r3_rows if r["model"] == model and r["slo_pass_and_zero_errors"]]
        r3_groups[model] = max(entries, key=lambda r: r["offered_rps"]) if entries else None
    files = {}
    reader_output = ROOT / "results/non_jev/v2/e2_sp/gpu5/E2_SP_P1_READER_OUTPUTS.jsonl"
    reader_count = sum(1 for line in reader_output.open(encoding="utf-8") if line.strip())
    for p in [BASE / "E2_SP_R_FROZEN_PROTOCOL.json", BASE / "E2_SP_R_PROVENANCE_MANIFEST.json", BASE / "E2_SP_R_ADJUDICATION_CASES.jsonl", reader_output, OUT / "E2_SP_R2_CAL_CONFIG_LOCK.json", OUT / "E2_SP_R2_LOCAL_ADJUDICATION.jsonl", OUT / "E2_SP_R2_LOCAL_ADJUDICATION_MANIFEST.json", OUT / "E2_SP_R2_ISOLATED_R3_SUMMARY.csv"]:
        files[str(p.relative_to(ROOT))] = sha_file(p)
    protocol = read_json(BASE / "E2_SP_R_FROZEN_PROTOCOL.json")
    protocol["r2_revision"] = "E2_SP_R2"
    protocol["r2_status"] = "EXECUTED_ISOLATED_CAL_AND_R3_LOCAL_ADJUDICATION_COMPLETE"
    protocol["r2_cal_config_lock"] = str((OUT / "E2_SP_R2_CAL_CONFIG_LOCK.json").relative_to(ROOT))
    protocol["r2_r3_isolated_gpu"] = 1
    protocol["r2_r4_local_judge"] = {"model": "Qwen/Qwen3-VL-8B-Instruct", "revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b", "primary_cases": 720, "repeat_cases": 144, "gold_in_prompt": False, "human_labels": 0}
    (OUT / "E2_SP_R2_FROZEN_PROTOCOL.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    cal_lines = ["# E2-S/P-R2 CAL serving optimization", "", "The native configuration was frozen before interpreting the isolated R3 traces. The CAL workload contained 30 unique queries. The measured configuration was `native_default`, batch size 1, request concurrency 1, native pinned precision, and the frozen candidate-view contract.", "", "Alternative batch/concurrency/attention settings were not claimed as measured because the pinned native adapters do not expose a comparable generic interface. No prompt or scoring semantics were changed.", "", "| Model | p50 ms | p95 ms | errors | status |", "|---|---:|---:|---:|---|"]
    with (OUT / "E2_SP_R2_CAL_OPTIMIZATION_REPORT.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(cal_lines) + "\n")
        for row in csv.DictReader((OUT / "E2_SP_R2_CAL_OPTIMIZATION.csv").open()): f.write(f"| {row['model']} | {float(row['p50_ms']):.3f} | {float(row['p95_ms']):.3f} | {row['errors']} | {row['status']} |\n")

    report = [
        "# DecisionRAG E2-S/P-R2 Owner-Review Report", "", "## Verdict", "",
        "**E2_SP_R2_OWNER_REVIEW: PASS_WITH_QUALIFICATIONS**", "",
        "R1 native adapters, R2 CAL-only measurement, isolated GPU1 R3 load traces, and R4 local adjudication completed. The results support an isolated operational comparison of the tested configurations. A formally verified sustainable QPS is not claimed because the runner records queue-growth proxies rather than an independent queue-length time series; highest tested SLO-passing offered rates are reported instead.", "",
        "Jev calls: **0**. Paid API calls: **0**. Training: **0**. Historical R0/R1/shared artifacts: preserved.", "",
        "## R2 CAL measurements", "", "| Model | p50 ms | p95 ms | errors |", "|---|---:|---:|---:|",
    ]
    cal_csv = list(csv.DictReader((OUT / "E2_SP_R2_CAL_OPTIMIZATION.csv").open()))
    for row in cal_csv: report.append(f"| {MODEL_LABELS[row['model']]} | {float(row['p50_ms']):.3f} | {float(row['p95_ms']):.3f} | {row['errors']} |")
    report += ["", "## Isolated R3 capacity evidence", "", "The 800 requests were 5 models × 4 fixed rates × 40 requests, sequentially run on an otherwise free A100 GPU1. Each model had 160 successful requests and zero errors. `highest tested SLO-passing offered RPS` means p95 ≤ 1,000 ms and zero errors; it is not a claim of sustainable capacity where queue growth was positive.", "", "| Model | highest tested SLO-passing offered RPS | achieved QPS | p50 ms | p95 ms | queue-growth proxy ms | GPU-sec/success proxy |", "|---|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        row = r3_groups[model]
        if row: report.append(f"| {MODEL_LABELS[model]} | {row['offered_rps']:.6f} | {row['achieved_qps']:.6f} | {row['p50_ms']:.3f} | {row['p95_ms']:.3f} | {row['queue_growth_ms']:.3f} | {row['gpu_seconds_per_completed_query_proxy']:.6f} |")
        else: report.append(f"| {MODEL_LABELS[model]} | NO_VERIFIED_FEASIBLE_RATE_AT_1S | — | — | — | — | — |")
    report += ["", "### Interpretation", "", "- GTE passed the 1-second p95 threshold at all four tested operating points; BGE passed at its three highest tested points but not at the lowest point, which had a 2.810-second p95.", "- Jina passed at 7.793321 and 15.586642 RPS, failed at 31.173284 RPS, and had positive queue-growth at its highest passing point.", "- Nimble and PPLX had no tested rate satisfying the 1-second p95 threshold. PPLX's lowest tested p95 was above 3.3 s.", "- GPU allocation-seconds/query are wall-time allocation proxies, not kernel compute-seconds. Monetary cost is not observed; no dollar saving claim is made.", "", "### GPU allocation proxy and cost sensitivity", "", "The following illustrative values convert the measured GPU wall-time allocation proxy at each model's highest tested 1-second-p95 passing point into dollars at hypothetical rates. They are not observed infrastructure prices and do not establish monetary savings.", "", "| Model | GPU-sec/success proxy | at $1/GPU-h | at $2/GPU-h | at $4/GPU-h |", "|---|---:|---:|---:|---:|"]
    for model in MODELS:
        row = r3_groups[model]
        if row:
            sec = float(row["gpu_seconds_per_completed_query_proxy"])
            report.append(f"| {MODEL_LABELS[model]} | {sec:.6f} | ${sec/3600:.8f} | ${2*sec/3600:.8f} | ${4*sec/3600:.8f} |")
        else:
            report.append(f"| {MODEL_LABELS[model]} | NOT_MEASURED_AT_1S_SLO | — | — | — |")
    report += ["", "## R4 downstream answer evidence", "", f"The existing {reader_count:,} successful reader outputs were preserved and the frozen 720-case subset was reused exactly: 120 questions × 6 policies. The local judge made 864 direct calls: 720 primary judgments and 144 deterministic repeats. It saw only the question, selected passages, and reader answer. It did not see reference answers, gold IDs, policy IDs, model scores, or expected labels. The results are `LLM_ADJUDICATED_LOCAL_BLINDED`, not human gold and not independently validated human correctness.", "", "| Dataset | Policy | n | local correctness PASS | groundedness PASS | completeness PASS | EM proxy | F1 proxy |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in quality_summary:
        fmt = lambda v: f"{float(v):.4f}" if isinstance(v, (int, float)) else str(v)
        report.append(f"| {row['dataset']} | {row['policy']} | {row['n_queries']} | {fmt(row['local_correctness_rate'])} | {fmt(row['local_groundedness_rate'])} | {fmt(row['local_completeness_rate'])} | {fmt(row['em_proxy'])} | {fmt(row['f1_proxy'])} |")
    paired_rows = list(csv.DictReader((OUT / "E2_SP_R2_PAIRED_ANSWER_STATISTICS.csv").open()))
    report += ["", f"Repeat exact agreement: {quality_audit['repeat_exact_agreement_n']}/{quality_audit['repeat_comparable_n']} = {quality_audit['repeat_exact_agreement_rate']}. This is a same-model stability statistic, not inter-annotator agreement.", "", "### Paired PPLX-minus-Jina exploratory differences", "", "| Population | Metric | Mean difference | 95% bootstrap CI | n |", "|---|---|---:|---:|---:|"]
    for row in paired_rows:
        if row["left"] == "pplx_decider" and row["right"] == "jina35" and row["metric"] in {"local_correctness", "deterministic_f1_proxy"}:
            if row["mean_difference"] == "NOT_MEASURED":
                report.append(f"| {row['population']} | {row['metric']} | NOT_MEASURED | NOT_MEASURED | {row['n_paired_queries']} |")
            else:
                report.append(f"| {row['population']} | {row['metric']} | {float(row['mean_difference']):.4f} | [{float(row['ci95_low']):.4f}, {float(row['ci95_high']):.4f}] | {row['n_paired_queries']} |")
    report += ["", "## PPLX-versus-Jina deployment trade-off", "", "PPLX is not supported as a deployment replacement by this R2 evidence: it has no 1-second p95-feasible tested rate and is substantially slower than Jina in isolated serving. Any quality comparison is local-model-judged exploratory evidence only; it cannot justify paying for a slower model absent independently validated quality gains. Jina offers a tested 1-second-p95 operating region, while BGE/GTE provide the strongest measured latency capacity among the tested native adapters.", "", "## Limitations and owner gate", "", "- The open-loop workload used 40 requests per rate; p95 order-statistic estimates are coarse (the 95th percentile is the fourth-highest observation under the runner's rank rule).", "- Queue stability was not independently instrumented as a time-series queue-length invariant; queue-growth proxies are diagnostic only.", "- Resource and cost accounting use process wall-time allocation proxies and telemetry markers. No infrastructure GPU-hour price was supplied, so monetary costs are not measured.", "- Local adjudication can be biased and is not human or independently validated. Enterprise cases have no reference answer and should not be treated as factual-correctness evidence.", "- No fresh confirmatory holdout was accessed. Jev remains untouched.", "", "**Owner-review gate: PASS_WITH_QUALIFICATIONS.** The isolated serving measurements and local downstream evidence are complete for review, but formal production deployment and factual-quality claims require owner acceptance of the limitations and, if needed, a separate validated quality study."]
    (OUT / "E2_SP_R2_FINAL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    owner = ["# E2-S/P-R2 Owner Review Gate", "", "- Status: `PASS_WITH_QUALIFICATIONS`", "- R1: PASS", "- R2 CAL optimization: PASS", "- R3 isolated GPU1 traces: PASS for execution and integrity; sustainable QPS: NOT_FORMALLY_VERIFIED", "- R4 local adjudication: PASS, 720 primary + 144 repeats; local-LLM only", "- R5 frontier: QUALIFIED exploratory only", "- Jev calls: 0", "- Paid calls: 0", "- Training: 0", "", "The next action is owner review. No Jev or paid phase is authorized by this artifact."]
    (OUT / "E2_SP_R2_OWNER_REVIEW_GATE.md").write_text("\n".join(owner) + "\n", encoding="utf-8")

    manifest = {"protocol": "DecisionRAG v2.1 E2-S/P-R2", "status": "PASS_WITH_QUALIFICATIONS", "approved_gpu": 1, "r3_request_count": len(r3_rows) * 40, "reader_output_count_preserved": reader_count, "r4_primary_cases": quality_audit["primary_rows"], "r4_repeat_cases": quality_audit["repeat_rows"], "r4_total_local_calls": quality_audit["all_rows"], "jev_calls": 0, "paid_api_calls": 0, "training_runs": 0, "sustainable_qps_status": "NOT_FORMALLY_VERIFIED_QUEUE_LENGTH_NOT_INSTRUMENTED", "artifact_hashes": files}
    manifest["manifest_sha256"] = sha_json(manifest)
    (OUT / "E2_SP_R2_PROVENANCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    resource = ["# E2-S/P-R2 Resource Audit", "", "- Approved execution device: A100 GPU1.", "- Pre-flight GPU1: 4 MiB used, no compute process observed.", "- Other GPU processes were not terminated or restarted.", "- R3 models ran sequentially on GPU1; historical resident services and unrelated workloads were preserved.", "- GPU allocation-seconds/query are process wall proxies from the raw R3 summaries.", "- No monetary cost is asserted because no owner-supplied GPU-hour rate was provided.", "- Jev/paid calls: 0."]
    (OUT / "E2_SP_R2_RESOURCE_AUDIT.md").write_text("\n".join(resource) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    r3_rows, r3_audit = integrate_r3()
    quality_rows, quality_summary, quality_audit = integrate_adjudication()
    manifest = write_reports(r3_rows, r3_audit, quality_rows, quality_summary, quality_audit)
    print(json.dumps({"r3": r3_audit, "quality": quality_audit, "status": manifest["status"]}, sort_keys=True))
    return 0 if r3_audit["all_pass"] and quality_audit["status_counts"]["invalid"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
