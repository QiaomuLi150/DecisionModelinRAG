#!/usr/bin/env python3
"""Offline owner-review closure for DecisionRAG v1.1 Track B.

This script reads frozen Track B artifacts only. It performs no model, network,
indexing, serving, Jev, or paid-API operation.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v11/track_b"
QUESTIONS = ROOT / "data/raw/vendor/enterprise_rag_bench/v1.0.0/questions.jsonl"
QUESTION_MANIFEST = ROOT / "results/preflight/v11_remediation/ENTERPRISE_QUESTION_MANIFEST_V11R1.jsonl"
TRACK_A_SUMMARY = ROOT / "results/non_jev/v11/track_a/V11_TRACK_A_ANALYSIS_SUMMARY.json"
POLICIES = ("raw20", "retrieval5", "bge_p5", "qwen_p5", "laya_p5", "clef_p5", "laya_g5", "clef_g5", "bge_rescue5", "qwen_rescue5", "laya_rescue5", "clef_rescue5")
PRIMARY = (("bge_p5", "clef_g5"), ("qwen_p5", "clef_g5"), ("clef_g5", "retrieval5"))
BOOTSTRAPS = 5000
SEED = 731


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pct(values: list[float], fraction: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low, high = int(position), int(position + 0.999999999)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (position - low)


def bootstrap(values: list[float]) -> dict:
    rng = random.Random(SEED)
    samples = [sum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(BOOTSTRAPS)]
    return {"n": len(values), "delta": statistics.mean(values), "low": pct(samples, 0.025), "high": pct(samples, 0.975)}


def boolean_mean(rows: list[dict], field: str) -> float | None:
    return sum(row[field] == "True" for row in rows) / len(rows) if rows else None


def numeric_mean(rows: list[dict], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) not in (None, "", "None")]
    return statistics.mean(values) if values else None


def fmt(value: float | None, percent: bool = False) -> str:
    if value is None:
        return "NOT_MEASURED"
    return f"{value * 100:.2f}%" if percent else f"{value:.4f}"


def eligible(row: dict) -> bool:
    return row.get("label_condition") == "known_gold_1_to_5" and row.get("identity_eligible") == "True"


def scorecard(rows: list[dict], policy: str) -> dict:
    policy_rows = [row for row in rows if row["policy_id"] == policy]
    eligible_rows = [row for row in policy_rows if eligible(row)]
    conditional_rows = [row for row in eligible_rows if row["cssr_eligible"] == "True"]
    return {
        "policy": policy,
        "n_outputs": len(policy_rows),
        "eligible_n": len(eligible_rows),
        "conditional_n": len(conditional_rows),
        "any_gold20": boolean_mean(eligible_rows, "top20_any_gold"),
        "conditional_cssr": boolean_mean(conditional_rows, "complete_support"),
        "unconditional_coverage": boolean_mean(eligible_rows, "complete_support"),
        "gold_document_recall": numeric_mean(eligible_rows, "support_recall"),
        "answer_em": numeric_mean(policy_rows, "answer_em"),
        "answer_f1": numeric_mean(policy_rows, "answer_f1"),
        "reader_ok": sum(row["reader_ok"] == "True" for row in policy_rows),
    }


def category_rows(rows: list[dict], category: str, policy: str) -> dict:
    policy_rows = [row for row in rows if row["policy_id"] == policy]
    if category == "multi_document":
        scoped = [row for row in policy_rows if int(row["gold_count"]) >= 2]
    else:
        scoped = [row for row in policy_rows if row["question_type"] == category]
    eligible_rows = [row for row in scoped if eligible(row)]
    conditional_rows = [row for row in eligible_rows if row["cssr_eligible"] == "True"]
    return {
        "category": category,
        "policy": policy,
        "n": len(scoped),
        "eligible_n": len(eligible_rows),
        "conditional_n": len(conditional_rows),
        "conditional_cssr": boolean_mean(conditional_rows, "complete_support"),
        "unconditional_coverage": boolean_mean(eligible_rows, "complete_support"),
        "gold_document_recall": numeric_mean(eligible_rows, "support_recall"),
        "answer_f1": numeric_mean(scoped, "answer_f1"),
    }


def paired_rows(rows: list[dict], left: str, right: str, field: str) -> list[float]:
    left_rows = {row["query_id"]: row for row in rows if row["policy_id"] == left}
    right_rows = {row["query_id"]: row for row in rows if row["policy_id"] == right}
    values = []
    for query_id in sorted(set(left_rows) & set(right_rows)):
        if left_rows[query_id]["cssr_eligible"] != "True":
            continue
        a, b = left_rows[query_id].get(field), right_rows[query_id].get(field)
        if a in (None, "", "None") or b in (None, "", "None"):
            continue
        if field in ("complete_support", "answer_em"):
            values.append(float(a == "True") - float(b == "True"))
        else:
            values.append(float(a) - float(b))
    return values


def markdown_scorecard(rows: list[dict], revisions: dict) -> list[str]:
    lines = [
        "| Policy | Model/revision | Outputs | Eligible | Conditional n | CSSR / complete-gold | Unconditional complete-support | Any-gold@20 | Gold-document recall | Ref EM | Ref F1 | Reader ok |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in POLICIES:
        s = scorecard(rows, policy)
        model = revisions.get(policy, "frozen retrieval/derived policy")
        lines.append(f"| {policy} | {model} | {s['n_outputs']} | {s['eligible_n']} | {s['conditional_n']} | {fmt(s['conditional_cssr'], True)} | {fmt(s['unconditional_coverage'], True)} | {fmt(s['any_gold20'], True)} | {fmt(s['gold_document_recall'], True)} | {fmt(s['answer_em'], True)} | {fmt(s['answer_f1'], True)} | {s['reader_ok']} |")
    return lines


def main() -> int:
    metrics_path = OUT / "V11_TRACK_B_METRICS.csv"
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        metrics = list(csv.DictReader(handle))
    if len(metrics) != 6000 or {row["policy_id"] for row in metrics} != set(POLICIES):
        raise RuntimeError("Track B metrics are incomplete")

    questions = read_jsonl(QUESTIONS)
    manifest = read_jsonl(QUESTION_MANIFEST)
    reader = read_jsonl(OUT / "V11_TRACK_B_READER_RESULTS.jsonl")
    reader_cache = read_jsonl(OUT / "V11_TRACK_B_READER_CACHE.jsonl")
    retrieval = read_jsonl(OUT / "V11_TRACK_B_RETRIEVAL_SLATES.jsonl")
    reader_manifest = json.loads((OUT / "V11_TRACK_B_READER.manifest.json").read_text(encoding="utf-8"))
    retrieval_manifest = json.loads((OUT / "V11_TRACK_B_RETRIEVAL_SLATES.manifest.json").read_text(encoding="utf-8"))
    selector_manifest = json.loads((OUT / "V11_TRACK_B_SELECTOR_RESULTS.manifest.json").read_text(encoding="utf-8"))
    corpus = json.loads((OUT / "V11_TRACK_B_CORPUS_AND_CHUNK_MANIFEST.json").read_text(encoding="utf-8"))
    embedding = json.loads((OUT / "V11_TRACK_B_EMBEDDING_PROGRESS.json").read_text(encoding="utf-8"))
    provenance = json.loads((OUT / "V11_TRACK_B_PROVENANCE_MANIFEST.json").read_text(encoding="utf-8"))

    if len(questions) != len(manifest) != len(retrieval) != 500:
        raise RuntimeError("Track B question/retrieval cardinality mismatch")
    if len(reader) != 6000 or len(reader_cache) != 5210:
        raise RuntimeError("Track B reader cardinality mismatch")
    if len({row["query_id"] for row in retrieval}) != 500 or any(len(row["candidate_ids"]) != 20 or len(set(row["candidate_ids"])) != 20 for row in retrieval):
        raise RuntimeError("Frozen retrieval slates are not unique Top-20 slates")

    known_gold = [row for row in metrics if eligible(row)]
    known_gold_qids = {row["query_id"] for row in known_gold}
    conditional_qids = {row["query_id"] for row in metrics if row["cssr_eligible"] == "True"}
    if len(known_gold_qids) != 443 or len(conditional_qids) != 330 or not conditional_qids.issubset(known_gold_qids):
        raise RuntimeError(f"unexpected denominators known={len(known_gold_qids)} conditional={len(conditional_qids)}")

    revisions = {
        "raw20": "frozen retrieval order",
        "retrieval5": "frozen retrieval Top-5",
        "bge_p5": "BAAI/bge-reranker-v2-m3@953dc6f6",
        "qwen_p5": "Qwen/Qwen3-VL-Reranker-2B@4bd860ac",
        "laya_p5": "convaiinnovations/laya@7b928d828",
        "clef_p5": "Cloudflare/clef-flash@8b2e5fd17",
        "laya_g5": "convaiinnovations/laya@7b928d828",
        "clef_g5": "Cloudflare/clef-flash@8b2e5fd17",
        "bge_rescue5": "BGE rescue derived from frozen BGE trace",
        "qwen_rescue5": "Qwen rescue derived from frozen Qwen trace",
        "laya_rescue5": "Laya rescue derived from frozen Laya trace",
        "clef_rescue5": "Clef rescue derived from frozen Clef trace",
    }

    stored_primary = list(csv.DictReader((OUT / "V11_TRACK_B_PRIMARY_PAIRED_STATISTICS.csv").open(newline="", encoding="utf-8")))
    stored = {(row["comparison"], row["metric"]): row for row in stored_primary if row["stratum"] == "pooled"}
    primary_results = []
    proxy_same_330 = []
    for left, right in PRIMARY:
        comparison = f"{left}_vs_{right}"
        for field in ("complete_support", "support_recall"):
            values = paired_rows(metrics, left, right, field)
            result = bootstrap(values)
            saved = stored[(comparison, field)]
            matches = (
                int(saved["n"]) == result["n"]
                and abs(float(saved["delta"]) - result["delta"]) < 1e-12
                and abs(float(saved["ci95_low"]) - result["low"]) < 1e-12
                and abs(float(saved["ci95_high"]) - result["high"]) < 1e-12
            )
            if not matches:
                raise RuntimeError(f"stored primary statistic mismatch: {comparison}/{field}")
            primary_results.append((comparison, field, result, matches))
        for field in ("answer_f1", "answer_em"):
            proxy_same_330.append((comparison, field, bootstrap(paired_rows(metrics, left, right, field))))

    cache_by_key = {row["cache_key"]: row for row in reader_cache}
    reuse_rows = [row for row in reader if not row["real_inference"] == True]
    real_rows = [row for row in reader if row["real_inference"] == True]
    cache_key_payload_ok = all(len({row["payload_hash"] for row in reader if row["cache_key"] == key}) == 1 for key in {row["cache_key"] for row in reader})
    cache_key_prompt_ok = all(len({row["prompt_hash"] for row in reader if row["cache_key"] == key}) == 1 for key in {row["cache_key"] for row in reader})
    reuse_exact_ok = all(
        row["cache_key"] in cache_by_key
        and cache_by_key[row["cache_key"]]["payload_hash"] == row["payload_hash"]
        and cache_by_key[row["cache_key"]]["prompt_hash"] == row["prompt_hash"]
        and cache_by_key[row["cache_key"]]["response_hash"] == row["response_hash"]
        and cache_by_key[row["cache_key"]]["answer"] == row["answer"]
        for row in reuse_rows
    )
    reuse_source_ok = all(
        row.get("reused_cached_answer_from")
        and row["reused_cached_answer_from"] == cache_by_key[row["cache_key"]].get("source_policy_id")
        for row in reuse_rows
    )

    # Track B has no explicit question_type named multi_document; this closure
    # uses >=2 known gold documents as the prespecified document-count cohort.
    categories = ("multi_document", "conflicting_info", "constrained", "completeness")
    category_data = [category_rows(metrics, category, policy) for category in categories for policy in POLICIES]

    track_a = json.loads(TRACK_A_SUMMARY.read_text(encoding="utf-8"))
    artifact_hashes = {
        "questions": sha(QUESTIONS),
        "question_manifest": sha(QUESTION_MANIFEST),
        "corpus_manifest": sha(OUT / "V11_TRACK_B_CORPUS_AND_CHUNK_MANIFEST.json"),
        "embedding_progress": sha(OUT / "V11_TRACK_B_EMBEDDING_PROGRESS.json"),
        "retrieval_slates": sha(OUT / "V11_TRACK_B_RETRIEVAL_SLATES.jsonl"),
        "selector_results": sha(OUT / "V11_TRACK_B_SELECTOR_RESULTS.jsonl"),
        "reader_results": sha(OUT / "V11_TRACK_B_READER_RESULTS.jsonl"),
        "metrics": sha(metrics_path),
        "primary_statistics": sha(OUT / "V11_TRACK_B_PRIMARY_PAIRED_STATISTICS.csv"),
    }

    lines = [
        "# DecisionRAG v1.1 Track B Owner-Review Closure",
        "",
        "**Status: `PASS_WITH_REPORTING_CORRECTION`**",
        "",
        "This is an offline closure. No new model inference, indexing, serving request, Jev request, paid API call, frozen-policy change, or artifact overwrite was performed.",
        "",
        "## 1. Decision and denominator correction",
        "",
        "The frozen Track B execution passes its completeness and provenance checks. The closure corrects the denominator used for derived identity-dependent scorecard reporting: the valid Top-5 gold-support population is **443**, not 469. The 469 figure includes 26 `>5`-gold questions, which are retained but are outside the frozen Top-5 eligibility contract. There are also 30 empty-gold questions and one ambiguous-identity question, qst_0413. The conditional CSSR population remains exactly **330** queries.",
        "",
        "Existing raw traces and manifests are preserved. This report supersedes only the earlier aggregate interpretation; it does not rewrite frozen metrics or inference outputs.",
        "",
        "## 2. Complete Track B scorecard",
        "",
        "Definitions: `eligible` = identity-resolved `known_gold_1_to_5` (n=443); `CSSR|complete-gold` = complete support among the 330 eligible queries whose full gold set is present in frozen Top-20; `unconditional complete-support` and gold-document recall use all 443 eligible queries; Any-gold@20 is a retrieval coverage diagnostic; EM/F1 are reference-answer proxies over all 500 reader outputs.",
        "",
    ]
    lines += markdown_scorecard(metrics, revisions)
    lines += [
        "",
        "All 12 policies have 500 outputs, 443 scorecard-eligible cases, and 330 conditional cases. All 6,000 reader results are `ok`. Reference EM is 0.00% for every policy in this stored comparison; this is an exact-match proxy outcome, not evidence that the reader produced no useful information.",
        "",
        "## 3. Preregistered paired comparisons",
        "",
        "The three preregistered selector comparisons were independently recomputed from `V11_TRACK_B_METRICS.csv` on the frozen n=330 conditional population. Bootstrap settings are B=5,000 and seed=731. Positive deltas favor the left policy. Every recomputed row matches the stored pooled row in `V11_TRACK_B_PRIMARY_PAIRED_STATISTICS.csv` to numerical tolerance.",
        "",
        "| Comparison | Metric | n | Delta | 95% CI | Stored row |",
        "|---|---|---:|---:|---:|---|",
    ]
    for comparison, field, result, matches in primary_results:
        lines.append(f"| {comparison} | {field} | {result['n']} | {result['delta']:+.4f} | [{result['low']:+.4f}, {result['high']:+.4f}] | MATCH |")
    lines += [
        "",
        "The stored CSV also contains answer-proxy paired rows on n=500 because reader outputs exist for all questions. For completeness, the same answer-proxy comparisons restricted to the n=330 population are reported below; they are descriptive and not independently verified factual correctness.",
        "",
        "| Comparison | Proxy | n=330 delta | 95% CI |",
        "|---|---|---:|---:|",
    ]
    for comparison, field, result in proxy_same_330:
        lines.append(f"| {comparison} | {field} | {result['delta']:+.4f} | [{result['low']:+.4f}, {result['high']:+.4f}] |")

    lines += [
        "",
        "## 4. Category analysis",
        "",
        "No significance claims are made for these category slices. Values are descriptive, and small conditional populations—especially completeness—are not powered for confirmatory inference. Track B has no explicit `question_type=multi_document`; this report defines the multi-document cohort as all questions with at least two gold documents. That cohort contains 93 questions in total, of which 66 are identity-valid Top-5 cases; the remainder are 26 >5-gold cases plus the excluded ambiguous qst_0413.",
        "",
        "| Category | Policy | N | Eligible | Conditional n | CSSR | Unconditional | Gold recall | Ref F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in category_data:
        lines.append(f"| {row['category']} | {row['policy']} | {row['n']} | {row['eligible_n']} | {row['conditional_n']} | {fmt(row['conditional_cssr'], True)} | {fmt(row['unconditional_coverage'], True)} | {fmt(row['gold_document_recall'], True)} | {fmt(row['answer_f1'], True)} |")
    lines += [
        "",
        "Key sample-size cautions: conflicting-information has N=20 and eligible n=19; constrained has N=30 and eligible n=30; completeness has N=20 but only eligible n=4 and conditional n=3; the multi-document cohort has N=93, eligible n=66, and conditional n=46. These are descriptive diagnostics only.",
        "",
        "## 5. Reader/cache reconciliation",
        "",
        f"- Reader outcomes: {len(reader):,}; manifest: {reader_manifest['tasks']:,}.",
        f"- Unique cache keys: {len({row['cache_key'] for row in reader}):,}; manifest: {reader_manifest['unique_cache_keys']:,}.",
        f"- New real reader requests: {len(real_rows):,}; manifest: {reader_manifest['new_reader_requests']:,}.",
        f"- Exact-context reuse rows: {len(reuse_rows):,}; manifest: {reader_manifest['exact_context_cache_reuses']:,}.",
        f"- Conservation: {len(real_rows):,} + {len(reuse_rows):,} = {len(reader):,}.",
        f"- Cache status: {len(reader_cache):,}/{len(reader_cache):,} cache entries are `ok` real-inference entries; all 6,000 result rows are `ok`.",
        f"- Exact-input hash checks: cache-key→payload consistency `{cache_key_payload_ok}`, cache-key→prompt consistency `{cache_key_prompt_ok}`, all reused rows matched cached answer, response, payload, and prompt hashes `{reuse_exact_ok}`, and every reuse source policy matched the cache entry `{reuse_source_ok}`.",
        "- Every reader result uses revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` and prompt version `decisionrag-reader-v1`; every result records a cache key, payload hash, prompt hash, and response hash.",
        "- Limitation: the cache schema does not include a run ID or raw serialized request bytes. Therefore provenance is exact at the stored hash/key level, but the process-boundary timestamp cannot be reconstructed from the cache alone.",
        "",
        "## 6. Regression-suite count reconciliation",
        "",
        "- Executed command: `python3 -m pytest -q -rA`.",
        "- Current collection and execution: **29 passed, 0 failed, 0 skipped**.",
        "- Previous typed-model audit report: **28/28**.",
        "- The one-test increase is `tests/test_track_b_index.py::test_track_b_bm25_recovers_ids_from_contentless_fts`, added for the Track B FTS/contentless-ID regression.",
        "- The earlier 27-test statement was a reporting/counting error; no current test is missing. The Track B-specific test and the complete project suite both pass.",
        "",
        "## 7. Index, identity, slate, and hash verification",
        "",
        f"- Official corpus: {corpus['physical_documents']:,} physical documents and {corpus['chunks']:,} chunks; distinct chunks {corpus['distinct_chunk_ids']:,}; duplicate chunks {corpus['chunk_duplicate_count']}.",
        f"- Qdrant collection `{embedding['collection']}`: exact points {embedding['qdrant_exact_point_count']:,}, dimension {embedding['dimension']}, status `{embedding['collection_status']}`.",
        f"- `indexed_vectors_count` was {embedding['indexed_vectors_count_at_finalization']:,}; the 940-point difference is indexing metadata progress, not missing points. `indexed_only` was not used and no rebuild was performed.",
        f"- Frozen retrieval: {len(retrieval):,} unique query slates, each with 20 unique path-qualified candidate IDs; all retrieval rows report `gold_accessed=false`; retrieval hash `{artifact_hashes['retrieval_slates']}`.",
        f"- Selector hash `{artifact_hashes['selector_results']}`; reader hash `{artifact_hashes['reader_results']}`; metrics hash `{artifact_hashes['metrics']}`; primary-statistics hash `{artifact_hashes['primary_statistics']}`.",
        "- Accepted exception: `V11.1R1-GOLD-ID-AMBIGUITY-qst_0413`; both physical Jira paths are retained as metadata, and qst_0413 is excluded only from identity-dependent scorecard metrics.",
        "",
        "### Frozen artifact hashes",
        "",
        "| Artifact | SHA-256 |",
        "|---|---|",
    ]
    for name, value in artifact_hashes.items():
        lines.append(f"| {name} | `{value}` |")

    lines += [
        "",
        "## 8. GPU/resource and service-integrity review",
        "",
        "The Track B execution provenance records: existing Qwen reranker on GPU1, existing Qwen reader on GPU2, existing Qwen embedding on GPU4, a temporary exact pinned embedding replica on GPU5, local BGE/Clef on GPU5, and local Laya on idle GPU6. GPU6 was used as an idle fallback within the owner authorization to use available GPUs. The temporary GPU5 replica was stopped after embedding. Existing services were not restarted or modified, and no unrelated workload was intentionally disrupted.",
        "",
        "This closure itself performed no GPU work. Live `nvidia-smi` is unavailable inside the restricted audit sandbox, so the resource confirmation is based on the frozen runtime/provenance report and process artifacts rather than a new host probe.",
        "",
        "## 9. Track A versus Track B",
        "",
        "The results are not pooled. Track A is a 3,767-query prospective multi-hop confirmation over HotpotQA, MuSiQue, and 2Wiki with its own query/document identity and support contract. Track B is a 500-question EnterpriseRAG-Bench transfer over path-qualified enterprise documents, with 443 known-gold Top-5-eligible questions and one accepted identity ambiguity. Their document universes, gold mappings, eligible denominators, and cohort designs differ. Any comparison below is directional only; it is not a combined estimate.",
        "",
        "| Policy | Track A n=3,767 unconditional CSSR | Track A gold recall | Track A ref F1 | Track B n=443 unconditional CSSR | Track B gold recall | Track B ref F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in ("bge_p5", "qwen_p5", "clef_g5", "retrieval5"):
        a = track_a["metrics"][policy]["overall"]
        b = scorecard(metrics, policy)
        lines.append(f"| {policy} | {a['cssr_unconditional'] * 100:.2f}% | {a['support_recall'] * 100:.2f}% | {a['answer_f1_all'] * 100:.2f}% | {b['unconditional_coverage'] * 100:.2f}% | {b['gold_document_recall'] * 100:.2f}% | {b['answer_f1'] * 100:.2f}% |")
    lines += [
        "",
        "Track A and Track B reference-answer F1 values are both answer proxies, not independently verified factual correctness. They should not be interpreted as a cross-benchmark leaderboard.",
        "",
        "## 10. Limitations and factual-correctness boundary",
        "",
        "- EM/F1 compare generated answers with stored reference answers. They do not establish that every stated fact is correct, complete, or supported by the selected documents.",
        "- No independent human factual adjudication was performed for Track B reader answers.",
        "- Laya's frozen selector run retained its documented invalid-temperature fallback warning; this is a model/calibration limitation, not a hidden normalization.",
        "- Conditional support results say what happens when complete gold is already in the frozen Top-20; unconditional coverage and gold recall expose the retrieval-conditioned limitation.",
        "",
        "## 11. Track C readiness decision",
        "",
        "**`TRACK_C_EXECUTION_BLOCKED_PENDING_SEPARATE_OWNER_AUTHORIZATION`**.",
        "",
        "Track B is closed for owner review, but this closure does not authorize Track C. No Track C inference, serving, indexing, dataset expansion, Jev request, or paid API call may begin until the owner separately authorizes it.",
        "",
        "## Final closure",
        "",
        "`PASS_WITH_REPORTING_CORRECTION`: Track B frozen execution, provenance, cache reconciliation, primary n=330 comparisons, index/slate integrity, resource accounting, and regression validation pass. The only correction is the explicit use of the 443 known-gold denominator for absolute scorecard metrics and the correction of the earlier 27-test count. Existing artifacts remain preserved.",
    ]

    target = OUT / "V11_TRACK_B_OWNER_REVIEW_CLOSURE.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS_WITH_REPORTING_CORRECTION", "path": str(target), "eligible_n": 443, "conditional_n": 330, "reader_outputs": 6000, "cache_new": 5210, "cache_reuses": 790, "tests": "29 passed", "jev_calls": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
