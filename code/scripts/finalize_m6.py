#!/usr/bin/env python3
"""Materialize the complete non-Jev owner report and final phase gate."""
from __future__ import annotations

import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ["raw20", "retrieval5", "bge_p5", "qwen_p5", "laya_p5", "clef_p5", "laya_g5", "clef_g5", "bge_rescue5", "qwen_rescue5", "laya_rescue5", "clef_rescue5"]
DATASETS = ["hotpot", "musique", "wiki2"]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value):
    return "NOT_MEASURED" if value is None else f"{100 * value:.2f}%"


def num(value, digits=2):
    return "NOT_MEASURED" if value is None else f"{value:.{digits}f}"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(headers: list[str], records: list[list[object]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out.extend("| " + " | ".join(str(value) for value in row) + " |" for row in records)
    return "\n".join(out)


def main() -> int:
    m4 = load_json(ROOT / "data/results/m4/m4_metrics.json")
    m5 = load_json(ROOT / "data/results/m5/m5_metrics.json")
    forecast = {
        "pricing_source": "https://typesafe.ai/blog/introducing-system-one-models-and-jev",
        "pricing_retrieved_utc": "2026-10-11",
        "input_price_usd_per_million_tokens": 0.042,
        "output_price": "FREE/too cheap to meter according to public TypeSafe announcement; verify account billing before approval",
        "rate_limits": "NOT_PUBLICLY_SPECIFIED; TypeSafe public enterprise page directs higher-limit customers to contact sales; verify account-specific limits during approved smoke",
        "smoke_calls_included": 0,
    }
    point_rows = [json.loads(line) for line in (ROOT / "data/results/m3/laya_all_pointwise.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    group_rows = [json.loads(line) for line in (ROOT / "data/results/m3/laya_all_groupwise.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    point_tokens = [item["input_tokens"] for row in point_rows for item in row["pointwise"]]
    group_tokens = [item["input_tokens"] for row in group_rows for item in row["groups"]]
    point_mean, group_mean = statistics.mean(point_tokens), statistics.mean(group_tokens)
    workloads = {
        "offline_cal": {"pointwise_calls": 150 * 20, "groupwise_calls": 150 * 5},
        "offline_test": {"pointwise_calls": 600 * 20, "groupwise_calls": 600 * 5},
        "live_serving": {"pointwise_calls": 100 * (20 + 20 + 20), "groupwise_calls": 100 * 5},
    }
    for item in workloads.values():
        item["total_calls"] = item["pointwise_calls"] + item["groupwise_calls"]
        item["estimated_input_tokens"] = item["pointwise_calls"] * point_mean + item["groupwise_calls"] * group_mean
        item["estimated_fee_usd"] = item["estimated_input_tokens"] / 1_000_000 * forecast["input_price_usd_per_million_tokens"]
    workloads["planned_total"] = {key: sum(item[key] for name, item in workloads.items() if name != "planned_total") for key in ("pointwise_calls", "groupwise_calls", "total_calls", "estimated_input_tokens", "estimated_fee_usd")}
    forecast.update({"observed_token_source": "M3 local Laya pointwise/groupwise traces using the frozen common views; Jev tokenizer/provider usage not yet observed", "pointwise_mean_input_tokens": point_mean, "groupwise_mean_input_tokens": group_mean, "workloads": workloads, "estimate_status": "ESTIMATED_NOT_BILLED"})
    (ROOT / "results/audit/jev_cost_forecast_offline.json").write_text(json.dumps(forecast, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    hash_paths = [
        "configs/study.yaml", "configs/models.yaml", "configs/prompts.json",
        "data/manifests/m1_data_manifest.json", "data/manifests/questions.jsonl", "data/manifests/split_manifest.jsonl", "data/manifests/gold_mapping_audit.jsonl",
        "data/corpus/passages.jsonl", "data/frozen_slates/m2_top20_slates.jsonl", "data/manifests/m2_retrieval_manifest.json", "data/results/m3/selection_results.jsonl", "data/manifests/m3_selection_manifest.json",
        "data/results/m4/reader_results.jsonl", "data/results/m4/m4_metrics.json", "data/manifests/m4_reader_manifest.json", "data/results/m5/m5_metrics.json", "data/results/m5/m5_topology_manifest.json",
    ]
    hashes = {path: sha256(ROOT / path) for path in hash_paths}
    (ROOT / "results/audit/hash_manifest.json").write_text(json.dumps({"study_id": "decisionrag-frozen-A4-B3-C4-D3-E3-F1-G2", "created_utc": datetime.now(timezone.utc).isoformat(), "sha256": hashes}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    quality_rows = []
    for policy in POLICIES:
        test = m4["metrics"][policy]["by_split"]["test"]
        label = "CSSR@20|gold_present" if policy == "raw20" else "CSSR@5|gold_present"
        quality_rows.append([policy, label, f"{test['gold_present_n']}/600", pct(test["cssr_gold_present"]), pct(test["cssr_unconditional"]), pct(test["support_recall"]), pct(test["answer_em"]), pct(test["answer_f1"]), num(test["reader_input_tokens_mean"], 1)])
    per_dataset_rows = []
    for row in m5["per_dataset"]:
        if row["policy_id"] in POLICIES:
            per_dataset_rows.append([row["policy_id"], row["dataset"], f"{row['cssr_numerator']}/{row['gold_present_n']}", pct(row["cssr_gold_present"]), pct(row["cssr_unconditional"]), pct(row["support_recall"]), pct(row["answer_em"]), pct(row["answer_f1"])])
    serving_rows = []
    for row in m5["summary"]:
        serving_rows.append([row["policy_id"], row["n_completed"], num(row["request_p50_ms"], 1), num(row["request_p95_ms"], 1), num(row["retrieval_ms_p50"], 1), num(row["selector_ms_p50"], 1), num(row["rescue_ms_p50"], 1), num(row["reader_ms_p50"], 1), num(row["reader_input_tokens_mean"], 1), row["error_count"], row["retry_count"]])
    ci_rows = []
    for row in m5["paired_ci"]:
        if row["scope"] == "equal_dataset_macro":
            n = ",".join(f"{key}:{value}" for key, value in row["n_by_dataset"].items())
        else:
            n = row.get("n", "")
        ci_rows.append([row["comparison"], row["scope"], n, pct(row.get("delta")), f"[{pct(row.get('ci95_low'))}, {pct(row.get('ci95_high'))}]", row["status"]])

    report = f"""# DecisionRAG Frozen Study — Complete Non-Jev Report

**Status: `NON_JEV_COMPLETE_JEV_PENDING_APPROVAL`**  
**Study:** `decisionrag-frozen-A4-B3-C4-D3-E3-F1-G2`  
**Report generated:** {datetime.now(timezone.utc).isoformat()}

This report closes M0–M6 of the authoritative SDD. All non-Jev experiments were executed with real data and real local/service inference. Jev was intentionally not contacted. The project is stopped at the required paid-API gate.

## Gate decision

| Gate | Status | Evidence |
|---|---|---|
| M0 audit/safety | DONE | `reports/M0_STATUS.md`; paid gate tests |
| M1 dataset/split/gold mapping | DONE | 750 selected questions; 20 valid mapping-audit examples per dataset |
| M2 hybrid retrieval/slate freeze | DONE | 750 immutable Top-20 slates; qrels not used during generation |
| M3 non-Jev selection | DONE | 12 policies × 750 = 9,000 real selection rows |
| M4 non-Jev reader | DONE | 12 policies × 750 = 9,000 accepted reader rows; all status `ok` |
| M5 non-Jev serving | DONE | 12 policies × 100 = 1,200 uncached end-to-end requests; concurrency 10 |
| M6 closure | DONE | 24/24 tests passed; tables, CIs, figures, hashes and this report |
| Jev paid gate | PENDING APPROVAL | 0 calls by design; no smoke or bulk approval supplied |

## Frozen data and provenance

- Datasets: HotpotQA, MuSiQue, 2WikiMultiHopQA; each has 50 CAL and 200 TEST questions, plus 10 debug fixtures.
- Corpus: 144,650 passages; all 750 selected gold mappings valid.
- Gold audit: 60 rows, exactly 20 per dataset, all valid. Twelve invalid source records were excluded before selection and are listed in the M1 manifest.
- Retrieval: BM25 top-100 + Qwen embedding top-100 + RRF(60), 20 unique candidates per query. Gold/qrels were not accessed during candidate generation.
- M2 gold presence after generation: HotpotQA 98.8%, MuSiQue 93.2%, 2Wiki 99.6% over the 250-row dataset pools.

Key SHA-256 values:

{table(['Artifact', 'SHA-256'], [[path, digest] for path, digest in hashes.items()])}

The complete machine-readable hash list is `results/audit/hash_manifest.json`.

## Non-Jev policy matrix and quality results

The 12 executed policies were `raw20`, `retrieval5`, BGE/Qwen/Laya/Clef pointwise `p5`, Laya/Clef groupwise `g5`, and four pointwise bridge-rescue variants. `raw20` is reported as CSSR@20; its conditional support retention is constructionally 100% when the full gold set is in Top-20 and is not evidence of a Top-5 selector.

All quality values below are over the complete 600-query TEST cohort. The conditional denominator is the policy-independent set of 291 TEST queries whose complete annotated gold set was present in the immutable Top-20; unconditional CSSR retains retrieval misses as failures.

{table(['Policy', 'Primary conditional metric', 'Denom.', 'Conditional', 'Unconditional', 'Support recall', 'EM', 'F1', 'Reader tokens'], quality_rows)}

### TEST dataset breakdown

{table(['Policy', 'Dataset', 'CSSR n/denom.', 'CSSR conditional', 'CSSR unconditional', 'Support recall', 'EM', 'F1'], per_dataset_rows)}

The raw per-query quality metrics are in `data/results/m4/m4_metrics.json`; the normalized table is `results/non_jev/analysis/per_dataset.csv` and `paper/tables/policy_quality.csv`.

### Paired bootstrap inference

The primary paired metric is complete-support retention conditional on gold presence. Bootstrap is query-paired within dataset, 5,000 replicates, seed 731. Comparisons with fewer than 40 eligible MuSiQue queries are marked `UNDERPOWERED_DESCRIPTIVE` as required.

{table(['Comparison', 'Scope', 'N', 'Delta', '95% CI', 'Status'], ci_rows)}

No noninferiority or universal-superiority claim is made. The strongest observed TEST quality result is local Clef groupwise conditional CSSR 52.58% (153/291), while local Clef is substantially slower in serving; this is an empirical result of this frozen benchmark, not a deployment recommendation by itself.

## M3 selector execution evidence

- BGE, Qwen, Laya and local Clef pointwise: 750/750 successful each.
- Laya and local Clef groupwise: 750/750 successful each, five disjoint quartets per query, corrected groupwise truncation 0/750.
- Rescue swaps: BGE 417, Qwen 418, Laya 423, Clef 424 over the 750-query offline run.
- Shared pointwise view hash mismatches: 0.
- Selection errors: 0/9,000. Hosted Cloudflare calls: 0. Jev calls: 0.
- The Laya checkpoint emitted the recorded invalid-temperature warning; its affected confidence values are treated as uncalibrated, not silently corrected.

## M4 reader execution evidence

- Reader: `Qwen/Qwen3-VL-8B-Instruct`, revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, qualified served model ID.
- Deterministic generation: temperature 0, top-p 1, seed 0, max 64 output tokens.
- Accepted results: 9,000/9,000 `ok`; 7,677 unique real reader requests and 1,323 exact-context cache reuses under the offline cache contract.
- The initial unqualified model identifier produced 404s; those failed attempts remain in cache history and were excluded from accepted metrics. The corrected qualified identifier was used for the accepted run; no model or service was changed.
- Reader TEST truncation audit: 7,200 policy/query rows; 1,924 had any passage truncation and 395 had an annotated supporting passage truncated. These are reported in `results/non_jev/analysis/truncation_audit.csv` and are not attributed to selector scoring.

## M5 complete-request serving results

Each policy ran 100 distinct TEST requests from the frozen cohort (HotpotQA 34, MuSiQue 33, 2Wiki 33), with fresh retrieval, fresh policy inference, rescue where applicable, and fresh reader generation. Response and selector-decision caches were disabled. p50/p95 are complete request durations, not sums of isolated p95 values.

{table(['Policy', 'n', 'Req p50 ms', 'Req p95 ms', 'Retrieval p50', 'Selector p50', 'Rescue p50', 'Reader p50', 'Reader tokens', 'Errors', 'Retries'], serving_rows)}

Full stage distributions and operational cost fields are in `results/non_jev/analysis/summary.csv` and `results/non_jev/analysis/operational_costs.csv`; paper-ready serving data are in `paper/tables/serving_cost.csv`.

Serving integrity: all 12 policies had 100/100 `ok`, 0 retries, 0 errors, `cache_disabled=true`, paid usage false, and provider fee 0.0. The topology hash is recorded in every trace and in `data/results/m5/m5_topology_manifest.json`.

### GPU and cost accounting

- M0 recorded eight NVIDIA A100-SXM4-80GB devices. Local selector windows used GPU0; existing model services were preserved and not restarted or modified.
- Trace peak allocation observations: BGE approximately 1.22 GiB; local Clef approximately 18.88 GiB. Service-backed per-request VRAM is not attributable from the client traces.
- Exact per-request GPU active seconds were **not instrumented** in the completed G2 traces. Therefore `GPU_seconds_per_1k` is explicitly `NOT_MEASURED`; the report does not call this zero. The separately reported `selector_wall_seconds_per_1k_proxy` is a measured wall-time proxy and must not be interpreted as GPU active time.
- Actual paid API fees for this non-Jev study are $0.00 because no Jev or hosted Cloudflare calls occurred. Local compute cost is `NOT_ESTIMATED` because no GPU-hour price was supplied or assumed. The operational cost CSV keeps paid fees separate from unpriced local compute.

## Tests and reproducibility

Final command:

```bash
./Production_Project/MultiModalRAG/venv/bin/python -m pytest -q
```

Result: **24 passed, 0 failed**. The suite covers paid-gate denial, data and split integrity, retrieval and qrel isolation, model-view alignment, groupwise limits, rescue invariants, reader metrics/cache, M5 cohort/stage/trace integrity, and stratified audit provenance.

Reproducible non-Jev commands are recorded in the phase reports and include:

```bash
python scripts/preflight.py --config configs/study.yaml --no-paid-api
python scripts/prepare_data.py --config configs/study.yaml
python scripts/m2_retrieve.py
python scripts/run_m3_selectors.py
python scripts/assemble_m3.py
python scripts/run_m4_reader.py
python scripts/analyze_m4.py
python scripts/run_m5_serving.py --policy <one-of-12-policies>
python scripts/repair_m5_metadata.py
python scripts/analyze_m5.py
python -m pytest -q
python scripts/finalize_m6.py
```

The parent worktree contained unrelated pre-existing changes; this study preserved them and confined its implementation/evidence to `Decision_Model_in_RAG`.

## Jev-only work remaining — not executed

Jev status: **`NOT_RUN_BY_DESIGN`**. Jev API calls: **0**. Jev charges: **$0.00**.

The frozen remaining workload is:

- Offline TEST: 600 × (20 pointwise + 5 groupwise) = **15,000 calls**.
- Offline CAL: 150 × (20 pointwise + 5 groupwise) = **3,750 calls**.
- Live G2: 100 × (20 `jev_p5` + 25 `jev_g5` including fresh pointwise ordering + 20 `jev_rescue5`) = **6,500 calls**.
- Planned total before separately approved smoke calls or charged retries: **25,250 calls**.

Offline-only fee forecast (not a bill and not an authorization):

- Public TypeSafe material currently states $0.042 per million input tokens ($42 per billion) and free output; the public API docs specify `POST /v1/systemone` but do not publish an account-specific rate limit. TypeSafe’s enterprise page directs higher-limit customers to contact sales.
- The estimate uses the observed frozen-view token distribution from real M3 Laya traces: pointwise mean {point_mean:.2f} input tokens/call; groupwise mean {group_mean:.2f} input tokens/call. Provider tokenizer usage and account billing are unverified until an approved smoke.
- Offline CAL: approximately {workloads['offline_cal']['estimated_input_tokens']:,.0f} input tokens, estimated **${workloads['offline_cal']['estimated_fee_usd']:.2f}**.
- Offline TEST: approximately {workloads['offline_test']['estimated_input_tokens']:,.0f} input tokens, estimated **${workloads['offline_test']['estimated_fee_usd']:.2f}**.
- Live G2: approximately {workloads['live_serving']['estimated_input_tokens']:,.0f} input tokens, estimated **${workloads['live_serving']['estimated_fee_usd']:.2f}**.
- Planned total: approximately {workloads['planned_total']['estimated_input_tokens']:,.0f} input tokens, estimated **${workloads['planned_total']['estimated_fee_usd']:.2f}**, excluding smoke calls and any billed retries. This forecast has a provider-tokenization and account-pricing uncertainty; it is not an owner-set budget ceiling.

The forecast is stored in `results/audit/jev_cost_forecast_offline.json`. No credential validation, model listing, health check, or inference request was sent to TypeSafe.

## Required stop and approval question

All non-Jev experiments are complete. Do you approve a limited paid Jev smoke test capped at **5 requests and $1.00 maximum additional spend**? I will not make any Jev request without that explicit approval. Bulk Jev inference will require a separate approval after the smoke report and revised forecast.
"""
    report_path = ROOT / "results/audit/NON_JEV_COMPLETE_REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    phase_state = {
        "study_id": "decisionrag-frozen-A4-B3-C4-D3-E3-F1-G2",
        "current_milestone": "M6",
        "status": "NON_JEV_COMPLETE_JEV_PENDING_APPROVAL",
        "milestones": {f"M{i}": "DONE" for i in range(7)},
        "non_jev_policies": 12,
        "non_jev_selection_rows": 9000,
        "non_jev_reader_rows": 9000,
        "non_jev_serving_requests": 1200,
        "tests_passed": 24,
        "tests_failed": 0,
        "jev_calls": 0,
        "jev_execution_allowed": False,
        "hosted_cloudflare_calls": 0,
        "jev_charges_usd": 0.0,
        "report": str(report_path),
        "forecast": str(ROOT / "results/audit/jev_cost_forecast_offline.json"),
        "hash_manifest": str(ROOT / "results/audit/hash_manifest.json"),
        "approval_required": {"smoke": True, "bulk": True},
    }
    (ROOT / "results/audit/phase_state.json").write_text(json.dumps(phase_state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": phase_state["status"], "report": str(report_path), "planned_jev_calls": workloads["planned_total"]["total_calls"], "estimated_planned_jev_fee_usd": workloads["planned_total"]["estimated_fee_usd"], "jev_calls": 0}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
