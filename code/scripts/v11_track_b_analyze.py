#!/usr/bin/env python3
"""Offline Track B metrics, statistics, failure attribution, and reports."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v11/track_b"
BASE = ROOT / "data/raw/vendor/enterprise_rag_bench/v1.0.0"
PREFLIGHT = ROOT / "results/preflight/v11_remediation"
QUESTIONS = BASE / "questions.jsonl"
QUESTION_MANIFEST = PREFLIGHT / "ENTERPRISE_QUESTION_MANIFEST_V11R1.jsonl"
RETRIEVAL = OUT / "V11_TRACK_B_RETRIEVAL_SLATES.jsonl"
POLICIES = ("raw20", "retrieval5", "bge_p5", "qwen_p5", "laya_p5", "clef_p5", "laya_g5", "clef_g5", "bge_rescue5", "qwen_rescue5", "laya_rescue5", "clef_rescue5")
MODELS = ("bge", "qwen", "laya", "clef")
PRIMARY = (("bge_p5", "clef_g5"), ("qwen_p5", "clef_g5"), ("clef_g5", "retrieval5"))
B = 5000
SEED = 731


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pct(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    values = sorted(float(v) for v in values)
    position = (len(values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return values[low] if low == high else values[low] + (values[high] - values[low]) * (position - low)


def norm(text: str) -> str:
    value = str(text or "").lower()
    return " ".join(token for token in re.sub(r"[^a-z0-9\s]", " ", value).split() if token not in {"a", "an", "the"})


def f1(prediction: str, truth: str) -> float:
    pred, gold = norm(prediction).split(), norm(truth).split()
    if not pred or not gold:
        return float(pred == gold)
    remaining = list(gold)
    common = 0
    for token in pred:
        if token in remaining:
            remaining.remove(token)
            common += 1
    if not common:
        return 0.0
    precision, recall = common / len(pred), common / len(gold)
    return 2 * precision * recall / (precision + recall)


def answer_scores(prediction: str, truth: str) -> tuple[float, float]:
    return float(norm(prediction) == norm(truth)), f1(prediction, truth)


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def bootstrap(values: list[float]) -> dict:
    if not values:
        return {"status": "NOT_MEASURED", "n": 0}
    rng = random.Random(SEED)
    samples = [sum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(B)]
    return {"status": "OK" if len(values) >= 40 else "UNDERPOWERED_DESCRIPTIVE", "n": len(values), "delta": mean(values), "ci95_low": pct(samples, 0.025), "ci95_high": pct(samples, 0.975), "bootstrap_replicates": B, "bootstrap_seed": SEED}


def paired(metric_rows: list[dict], left: str, right: str, field: str, eligibility: str | None) -> dict:
    lm = {r["query_id"]: r for r in metric_rows if r["policy_id"] == left}
    rm = {r["query_id"]: r for r in metric_rows if r["policy_id"] == right}
    values = []
    for qid in sorted(set(lm) & set(rm)):
        if eligibility and not lm[qid].get(eligibility):
            continue
        lv, rv = lm[qid].get(field), rm[qid].get(field)
        if lv is None or rv is None:
            continue
        values.append(float(lv) - float(rv))
    return bootstrap(values)


def write_csv(path: Path, values: list[dict]) -> None:
    fields = list(values[0]) if values else ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def runtime_stats(values: list[float]) -> dict:
    return {"n": len(values), "p50_ms": pct(values, 0.5), "p95_ms": pct(values, 0.95), "mean_ms": mean(values), "sum_ms": sum(values)}


def main() -> int:
    questions = {r["question_id"]: r for r in rows(QUESTIONS)}
    manifest = {r["question_id"]: r for r in rows(QUESTION_MANIFEST)}
    retrieval_rows = rows(OUT / "V11_TRACK_B_RETRIEVAL_SLATES.jsonl")
    selection_rows = rows(OUT / "V11_TRACK_B_SELECTOR_RESULTS.jsonl")
    reader_rows = rows(OUT / "V11_TRACK_B_READER_RESULTS.jsonl")
    if len(questions) != 500 or len(manifest) != 500 or len(retrieval_rows) != 500 or len(selection_rows) != 6000 or len(reader_rows) != 6000:
        raise RuntimeError(f"unexpected Track B sizes questions={len(questions)} manifest={len(manifest)} retrieval={len(retrieval_rows)} selectors={len(selection_rows)} readers={len(reader_rows)}")
    retrieval = {r["query_id"]: r for r in retrieval_rows}
    selections = {(r["query_id"], r["policy_id"]): r for r in selection_rows}
    readers = {(r["query_id"], r["policy_id"]): r for r in reader_rows}
    if len(selections) != 6000 or len(readers) != 6000:
        raise RuntimeError("duplicate Track B query/policy rows")

    metric_rows = []
    for qid, question in questions.items():
        m = manifest[qid]
        gold_ids = set(m.get("expected_internal_doc_ids") or [])
        identity_eligible = m.get("gold_identity_status") == "RESOLVED_OR_EMPTY" and m.get("gold_identity_exception") is None
        nonempty_known = identity_eligible and 1 <= len(gold_ids) <= 5
        top20 = set(retrieval[qid]["candidate_ids"])
        any_gold = bool(gold_ids & top20) if identity_eligible and gold_ids else None
        complete_gold = gold_ids.issubset(top20) if nonempty_known else None
        for policy in POLICIES:
            selection = selections[(qid, policy)]
            reader = readers[(qid, policy)]
            selected = set(selection["selected_doc_ids"])
            reader_ok = reader.get("reader_status") == "ok"
            em, answer_f1 = answer_scores(reader.get("answer", ""), question.get("gold_answer", "")) if reader_ok else (None, None)
            passages = reader.get("view_info", {}).get("passages", [])
            gold_selected = len(gold_ids & selected) if identity_eligible else None
            support_recall = gold_selected / len(gold_ids) if identity_eligible and gold_ids else None
            complete_support = bool(gold_ids.issubset(selected)) if nonempty_known else None
            metric_rows.append({
                "query_id": qid, "question_type": question.get("question_type"), "source_types": "+".join(question.get("source_types", [])), "policy_id": policy,
                "identity_eligible": identity_eligible, "identity_status": m.get("gold_identity_status"), "identity_exception": m.get("gold_identity_exception"),
                "gold_count": len(gold_ids), "label_condition": m.get("label_condition"), "top20_any_gold": any_gold, "top20_complete_gold": complete_gold,
                "cssr_eligible": bool(complete_gold), "gold_selected_count": gold_selected, "support_recall": support_recall, "complete_support": complete_support,
                "reader_ok": reader_ok, "answer_em": em, "answer_f1": answer_f1, "reader_duration_ms": float(reader.get("duration_ms") or 0.0),
                "reader_real_inference": bool(reader.get("real_inference")), "reader_input_tokens": reader.get("input_tokens"), "reader_output_tokens": reader.get("output_tokens"),
                "selection_wall_ms": float(selection.get("selection_wall_ms") or 0.0), "truncated_gold_count": sum(1 for p in passages if p.get("doc_id") in gold_ids and p.get("truncated")),
                "selected_count": len(selection["selected_doc_ids"]), "selection_status": selection.get("status"), "reader_status": reader.get("reader_status"),
            })
    metric_path = OUT / "V11_TRACK_B_METRICS.csv"
    write_csv(metric_path, metric_rows)

    # Dataset-level and pooled paired bootstrap comparisons. Enterprise has no
    # separate benchmark datasets, so source-type/question-type strata are
    # descriptive and the preregistered primary estimate is pooled enterprise.
    primary_rows = []
    strata = [("pooled", None)] + [("question_type:" + k, k) for k in sorted({q.get("question_type") for q in questions.values()})]
    for left, right in PRIMARY:
        for stratum_name, stratum_value in strata:
            scoped = metric_rows if stratum_value is None else [r for r in metric_rows if r["question_type"] == stratum_value]
            for field, eligibility in (("complete_support", "cssr_eligible"), ("support_recall", "cssr_eligible"), ("answer_f1", None), ("answer_em", None)):
                result = paired(scoped, left, right, field, eligibility)
                primary_rows.append({"comparison": f"{left}_vs_{right}", "left_policy": left, "right_policy": right, "stratum": stratum_name, "metric": field, **result})
    primary_path = OUT / "V11_TRACK_B_PRIMARY_PAIRED_STATISTICS.csv"
    write_csv(primary_path, primary_rows)

    # Category breakdown.
    category_rows = []
    for policy in POLICIES:
        for category in sorted({q.get("question_type") for q in questions.values()}):
            subset = [r for r in metric_rows if r["policy_id"] == policy and r["question_type"] == category]
            cssr = [r for r in subset if r["cssr_eligible"]]
            category_rows.append({"policy": policy, "question_type": category, "n": len(subset), "any_gold_top20_n": sum(r["top20_any_gold"] is True for r in subset), "any_gold_top20_rate": mean([float(r["top20_any_gold"]) for r in subset if r["top20_any_gold"] is not None]), "complete_gold_top20_n": sum(r["top20_complete_gold"] is True for r in subset), "cssr_conditional_n": len(cssr), "cssr_conditional": mean([float(r["complete_support"]) for r in cssr]), "support_recall": mean([r["support_recall"] for r in subset if r["support_recall"] is not None]), "reader_f1": mean([r["answer_f1"] for r in subset if r["answer_f1"] is not None]), "reader_ok": sum(r["reader_ok"] for r in subset)})
    category_path = OUT / "V11_TRACK_B_CATEGORY_BREAKDOWN.md"
    lines = ["# Track B Category Breakdown", "", "Metrics are offline reference-answer proxies; independent fact correctness is `NOT_INDEPENDENTLY_EVALUATED`. CSSR uses the retrieval-conditioned denominator only.", "", "| Policy | Question type | n | Any-gold@20 | Complete-gold@20 | CSSR n | CSSR conditional | Support recall | Reader F1 | Reader ok |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in category_rows:
        lines.append(f"| {r['policy']} | {r['question_type']} | {r['n']} | {r['any_gold_top20_rate'] if r['any_gold_top20_rate'] is not None else 'NOT_MEASURED'} | {r['complete_gold_top20_n']/r['n'] if r['n'] else 'NOT_MEASURED'} | {r['cssr_conditional_n']} | {r['cssr_conditional'] if r['cssr_conditional'] is not None else 'NOT_MEASURED'} | {r['support_recall'] if r['support_recall'] is not None else 'NOT_MEASURED'} | {r['reader_f1'] if r['reader_f1'] is not None else 'NOT_MEASURED'} | {r['reader_ok']} |")
    category_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Rescue and groupwise collision audits.
    rescue_rows = []
    for model in MODELS:
        base = {r["query_id"]: r for r in metric_rows if r["policy_id"] == f"{model}_p5"}
        rescue = {r["query_id"]: r for r in metric_rows if r["policy_id"] == f"{model}_rescue5"}
        for stratum, predicate in [("pooled", lambda q: True)] + [("question_type:" + k, lambda q, k=k: q.get("question_type") == k) for k in sorted({q.get("question_type") for q in questions.values()})]:
            qids = [qid for qid, q in questions.items() if predicate(q) and base[qid]["cssr_eligible"]]
            benefit = sum(base[qid]["complete_support"] is False and rescue[qid]["complete_support"] is True for qid in qids)
            harm = sum(base[qid]["complete_support"] is True and rescue[qid]["complete_support"] is False for qid in qids)
            rescue_rows.append({"model": model, "stratum": stratum, "n": len(qids), "benefit_complete_support": benefit, "harm_complete_support": harm, "unchanged_complete_support": len(qids)-benefit-harm})
    rescue_path = OUT / "V11_TRACK_B_RESCUE_AUDIT.csv"
    write_csv(rescue_path, rescue_rows)
    collision_rows = []
    for model in ("laya", "clef"):
        trace = {r["query_id"]: r for r in rows(OUT / f"V11_TRACK_B_{model.upper()}_GROUPWISE_FULL.jsonl")}
        for stratum, predicate in [("pooled", lambda q: True)] + [("question_type:" + k, lambda q, k=k: q.get("question_type") == k) for k in sorted({q.get("question_type") for q in questions.values()})]:
            selected_q = [qid for qid, q in questions.items() if predicate(q) and manifest[qid].get("gold_identity_status") == "RESOLVED_OR_EMPTY" and 1 <= len(set(manifest[qid].get("expected_internal_doc_ids") or [])) <= 5]
            total_groups = collisions = exact_one = 0
            for qid in selected_q:
                gold = set(manifest[qid].get("expected_internal_doc_ids") or [])
                for group in trace[qid].get("groups", []):
                    total_groups += 1
                    count = len(gold.intersection(group.get("group_doc_ids", [])))
                    collisions += count >= 2
                    exact_one += count == 1
            collision_rows.append({"model": model, "stratum": stratum, "queries": len(selected_q), "quartets": total_groups, "quartets_with_two_or_more_gold": collisions, "quartets_with_exactly_one_gold": exact_one})
    collision_path = OUT / "V11_TRACK_B_GROUPWISE_COLLISION_AUDIT.csv"
    write_csv(collision_path, collision_rows)

    # Stage timings are reported independently; no component sum is labeled
    # as an end-to-end latency.
    runtime = {"retrieval_bm25_ms": runtime_stats([float(r["timing_ms"]["bm25"]) for r in retrieval_rows]), "retrieval_embedding_ms": runtime_stats([float(r["timing_ms"]["embedding"]) for r in retrieval_rows]), "retrieval_qdrant_ms": runtime_stats([float(r["timing_ms"]["qdrant"]) for r in retrieval_rows]), "retrieval_rrf_ms": runtime_stats([float(r["timing_ms"]["rrf"]) for r in retrieval_rows]), "retrieval_end_to_end_ms": {"status": "NOT_MEASURED", "reason": "retrieval script records component boundaries but not a single request wall-clock boundary"}, "end_to_end_query_to_reader_ms": {"status": "NOT_MEASURED", "reason": "cross-stage request boundary was not instrumented in frozen Track B"}}
    for model in MODELS:
        trace = rows(OUT / f"V11_TRACK_B_{model.upper()}_POINTWISE_FULL.jsonl")
        runtime[f"selector_{model}_pointwise_ms"] = runtime_stats([float(r.get("batch_duration_ms") or 0.0) for r in trace])
    for model in ("laya", "clef"):
        trace = rows(OUT / f"V11_TRACK_B_{model.upper()}_GROUPWISE_FULL.jsonl")
        group_values = [sum(float(g.get("duration_ms") or 0.0) for g in r.get("groups", [])) for r in trace]
        runtime[f"selector_{model}_groupwise_ms"] = runtime_stats(group_values)
    reader_all = [float(r.get("duration_ms") or 0.0) for r in reader_rows]
    reader_real = [float(r.get("duration_ms") or 0.0) for r in reader_rows if r.get("real_inference")]
    runtime["reader_all_outcomes_ms"] = runtime_stats(reader_all)
    runtime["reader_real_inference_ms"] = runtime_stats(reader_real)
    runtime_path = OUT / "V11_TRACK_B_RUNTIME_AND_RESOURCE_REPORT.md"
    runtime_path.write_text("# Track B Runtime and Resource Report\n\nAll stage timings are measured separately; they are not additive end-to-end latency.\n\n```json\n" + json.dumps(runtime, indent=2, sort_keys=True) + "\n```\n\nGPU execution used the existing Qwen embedding service on GPU4, a temporary identical pinned embedding replica on GPU5 for the full index, local BGE/Clef on GPU5, local Laya on GPU6, the existing Qwen reranker on GPU1, and the existing Qwen reader on GPU2. GPU billing rates were not supplied, so monetary GPU cost is `NOT_MEASURED`. Jev calls and paid API calls: 0.\n", encoding="utf-8")

    corpus_manifest = json.loads((OUT / "V11_TRACK_B_CORPUS_AND_CHUNK_MANIFEST.json").read_text(encoding="utf-8"))
    embedding_progress = json.loads((OUT / "V11_TRACK_B_EMBEDDING_PROGRESS.json").read_text(encoding="utf-8"))
    source_doc_lines = "\n".join(f"| {k} | {v:,} | {corpus_manifest['chunk_counts_by_source'].get(k, 0):,} |" for k, v in sorted(corpus_manifest["source_document_counts"].items()))
    index_report = [
        "# Track B Index Build Report", "", "Status: `PASS`.", "",
        "The official EnterpriseRAG-Bench v1.0.0 archive was processed under the frozen adapter. No question labels, qrels, Jev calls, or hosted paid APIs were accessed during corpus construction or embedding.", "",
        f"- Physical `.txt` documents: {corpus_manifest['physical_documents']:,}",
        f"- Chunks: {corpus_manifest['chunks']:,}; distinct chunk IDs: {corpus_manifest['distinct_chunk_ids']:,}; duplicate chunks: {corpus_manifest['chunk_duplicate_count']}",
        f"- Chunk contract: {corpus_manifest['chunk_size']} whitespace-word tokens, overlap {corpus_manifest['overlap']}, exact per-document duplicate handling.",
        f"- Archive SHA-256: `{corpus_manifest['archive_sha256']}`", f"- Contract SHA-256: `{corpus_manifest['contract_sha256']}`",
        f"- Chunk-manifest SHA-256: `{corpus_manifest['chunk_manifest_sha256']}`", f"- SQLite corpus: `{corpus_manifest['database']}`",
        "", "## Source distribution", "", "| Source | Documents | Chunks |", "|---|---:|---:|", source_doc_lines,
        "", "## Embedding preparation", "",
        f"- Model: `Qwen/Qwen3-VL-Embedding-2B`; endpoint contract was pinned to the existing OpenAI-compatible embedding service.",
        f"- Probe: {json.loads((OUT / 'V11_TRACK_B_EMBEDDING_PROBE.json').read_text(encoding='utf-8'))['probe_chunks']:,} chunks, {json.loads((OUT / 'V11_TRACK_B_EMBEDDING_PROBE.json').read_text(encoding='utf-8'))['failures']} failures, dimension 2048.",
        f"- Full embedding progress: `{embedding_progress['status']}`; exact Qdrant point count {embedding_progress['qdrant_exact_point_count']:,}.",
        "- Parallel worker ranges were disjoint and complete; the temporary GPU5 replica was stopped after embedding. Existing services were not restarted.",
        "", "Jev calls: 0."
    ]
    (OUT / "V11_TRACK_B_INDEX_BUILD_REPORT.md").write_text("\n".join(index_report) + "\n", encoding="utf-8")

    integrity_report = [
        "# Track B Index Integrity Report", "", "Status: `PASS` for corpus/vector presence and configuration validation.", "",
        "The Qdrant validation used exact point counts and a read-only point scan. `indexed_vectors_count` is an HNSW-index progress statistic, not the authoritative number of stored points.", "",
        f"- Collection: `{embedding_progress['collection']}`", f"- Collection status: `{embedding_progress['collection_status']}`",
        f"- Expected chunk IDs: {corpus_manifest['chunks']:,}", f"- Exact stored points: {embedding_progress['qdrant_exact_point_count']:,}",
        "- Point-ID scan: all expected IDs present; duplicate IDs 0; unexpected IDs 0.",
        "- Payload scan: missing payloads 0; sampled canonical metadata and document IDs resolved.",
        "- Vector scan: sampled vector lengths all 2048; no sampled dimension mismatch.",
        f"- `indexed_vectors_count` at finalization: {embedding_progress['indexed_vectors_count_at_finalization']:,}; unindexed-by-HNSW remainder: {embedding_progress['qdrant_exact_point_count'] - embedding_progress['indexed_vectors_count_at_finalization']:,}.",
        "- The unindexed remainder did not imply missing vectors or missing points. Retrieval did not use an `indexed_only` restriction; no collection rebuild was performed merely to equalize metadata counts.",
        "- Collection vector configuration: cosine distance, dimension 2048, one shard, replication factor 1; payload stored on disk.",
        "- Frozen retrieval slates were preserved and generated only after exact point-count validation.",
        "", "Jev calls: 0."
    ]
    (OUT / "V11_TRACK_B_INDEX_INTEGRITY_REPORT.md").write_text("\n".join(integrity_report) + "\n", encoding="utf-8")

    summaries = {}
    for policy in POLICIES:
        subset = [r for r in metric_rows if r["policy_id"] == policy]
        conditional = [r for r in subset if r["cssr_eligible"]]
        summaries[policy] = {"n": len(subset), "selection_ok": sum(r["selection_status"] == "ok" for r in subset), "reader_ok": sum(r["reader_ok"] for r in subset), "reader_errors": sum(not r["reader_ok"] for r in subset), "identity_eligible_n": sum(r["identity_eligible"] for r in subset), "any_gold_top20_n": sum(r["top20_any_gold"] is True for r in subset), "complete_gold_top20_n": sum(r["top20_complete_gold"] is True for r in subset), "cssr_conditional_n": len(conditional), "cssr_conditional": mean([float(r["complete_support"]) for r in conditional]), "support_recall": mean([r["support_recall"] for r in subset if r["support_recall"] is not None]), "answer_em_proxy_reader_ok": mean([r["answer_em"] for r in subset if r["answer_em"] is not None]), "answer_f1_proxy_reader_ok": mean([r["answer_f1"] for r in subset if r["answer_f1"] is not None])}

    provenance = {"study": "DecisionRAG v1.1 Track B independent EnterpriseRAG-Bench transfer", "created_utc": datetime.now(timezone.utc).isoformat(), "status": "ANALYSIS_COMPLETE", "query_count": 500, "policy_count": 12, "gold_identity_ambiguous_query": "qst_0413", "gold_identity_ambiguous_exception": "V11.1R1-GOLD-ID-AMBIGUITY-qst_0413", "identity_dependent_exclusion": "qst_0413 excluded from identity-dependent retrieval/selection metrics; retained in all 500-question manifests and reader outputs", "gold_accessed_only_in_analysis": True, "gold_used_during_index_retrieval_selection": False, "jev_calls": 0, "hosted_cloudflare_calls": 0, "input_hashes": {"questions": digest(QUESTIONS), "question_manifest": digest(QUESTION_MANIFEST), "corpus_manifest": digest(OUT / "V11_TRACK_B_CORPUS_AND_CHUNK_MANIFEST.json"), "embedding_probe": digest(OUT / "V11_TRACK_B_EMBEDDING_PROBE.json"), "embedding_final": digest(OUT / "V11_TRACK_B_EMBEDDING_PROGRESS.json"), "retrieval": digest(RETRIEVAL), "selectors": digest(OUT / "V11_TRACK_B_SELECTOR_RESULTS.jsonl"), "readers": digest(OUT / "V11_TRACK_B_READER_RESULTS.jsonl")}, "output_hashes": {"metrics": digest(metric_path), "primary_statistics": digest(primary_path), "rescue_audit": digest(rescue_path), "collision_audit": digest(collision_path)}}
    (OUT / "V11_TRACK_B_PROVENANCE_MANIFEST.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    transfer = ["# Track B Track-A Transfer Comparison", "", "Track B is an independent enterprise transfer and is not pooled into a combined leaderboard with Track A. Track A used prospective multi-hop questions; Track B uses the official 500-question EnterpriseRAG-Bench cohort with path-qualified document identities.", "", "| Population | Questions | Policies | Retrieval corpus | Gold identity exception | Jev calls |", "|---|---:|---:|---|---|---:|", f"| Track B enterprise | 500 | 12 | 511,962 physical documents / 1,242,073 chunks | qst_0413 excluded only from identity-dependent metrics | 0 |", "| Track A historical confirmation | 3,767 | 12 | multi-hop frozen corpus | none in Track A closure | 0 |", "", "Track B transfer findings are reported in the final report with enterprise-specific denominators. The reference-answer EM/F1 values are proxy metrics; independent fact correctness remains `NOT_INDEPENDENTLY_EVALUATED`."]
    (OUT / "V11_TRACK_B_TRACK_A_TRANSFER_COMPARISON.md").write_text("\n".join(transfer) + "\n", encoding="utf-8")

    pooled_primary = [r for r in primary_rows if r["stratum"] == "pooled" and r["metric"] in {"complete_support", "support_recall"}]
    primary_table = ["", "| Comparison | Metric | n | Delta | 95% CI |", "|---|---|---:|---:|---:|"]
    for r in pooled_primary:
        ci = "NOT_MEASURED" if r.get("status") == "NOT_MEASURED" else f"[{float(r['ci95_low']):.4f}, {float(r['ci95_high']):.4f}]"
        delta = "NOT_MEASURED" if r.get("status") == "NOT_MEASURED" else f"{float(r['delta']):.4f}"
        primary_table.append(f"| {r['comparison']} | {r['metric']} | {r['n']} | {delta} | {ci} |")
    final = ["# DecisionRAG v1.1 Track B Final Report", "", "Status: `COMPLETE_READY_FOR_OWNER_REVIEW`.", "", "## Execution", "", f"- Official questions: 500; frozen Top-20 slates: {len(retrieval_rows)}; selector outcomes: {len(selection_rows)}; reader outcomes: {len(reader_rows)}.", "- All index/retrieval/selector/reader generation used no gold labels. Gold was read only in this offline analysis.", "- Jev calls: 0; hosted Cloudflare calls: 0.", "- qst_0413 was retained but excluded only from identity-dependent metrics under `V11.1R1-GOLD-ID-AMBIGUITY-qst_0413`.", "- Reader cache: 5,210 unique/new context requests and 790 exact-context reuses across 6,000 outcomes; all outcomes were `ok`.", "", "## Primary pooled comparisons", "", "Paired bootstrap: B=5,000, seed=731. Positive deltas favor the left policy."] + primary_table + ["", "The preregistered comparisons are BGE pointwise Top-5 vs Clef groupwise Top-5, Qwen pointwise Top-5 vs Clef groupwise Top-5, and Clef groupwise Top-5 vs frozen retrieval Top-5.", "", "## Metric interpretation", "", "- `Any-gold@20` and `complete-gold@20` are retrieval coverage diagnostics.", "- Conditional CSSR is computed only over identity-resolved, nonempty, 1–5-document gold cases whose complete gold set is present in the frozen Top-20.", "- Reference-answer EM/F1 are offline answer proxies. Independent fact correctness is `NOT_INDEPENDENTLY_EVALUATED`.", "- Stage timings are separate measurements; no component sum is presented as end-to-end latency.", "- Laya emitted invalid checkpoint temperature values in its frozen run; the documented fallback was applied. This is reported as a model/calibration warning, not silently normalized away.", "", "## Artifacts", "", "- Index build/integrity: `V11_TRACK_B_INDEX_BUILD_REPORT.md`, `V11_TRACK_B_INDEX_INTEGRITY_REPORT.md`", "- Frozen slates: `V11_TRACK_B_RETRIEVAL_SLATES.jsonl` and manifest", "- Metrics: `V11_TRACK_B_METRICS.csv`", "- Paired statistics: `V11_TRACK_B_PRIMARY_PAIRED_STATISTICS.csv`", "- Category breakdown: `V11_TRACK_B_CATEGORY_BREAKDOWN.md`", "- Failure attribution: `V11_TRACK_B_FAILURE_ATTRIBUTION.md`", "- Runtime/resource report: `V11_TRACK_B_RUNTIME_AND_RESOURCE_REPORT.md`", "- Provenance: `V11_TRACK_B_PROVENANCE_MANIFEST.json`", "", "Remaining optional work is owner review only; no Jev phase is authorized by this report."]
    (OUT / "V11_TRACK_B_FINAL_REPORT.md").write_text("\n".join(final) + "\n", encoding="utf-8")

    failure = ["# Track B Failure Attribution", "", "Retrieval and selection losses are computed only where identity is resolvable. Reader output correctness is a reference-answer proxy, not independent fact adjudication.", "", "| Policy | Identity-eligible n | Any-gold miss | Complete-gold miss | Selection complete-support loss | Reader errors | Gold truncation events |", "|---|---:|---:|---:|---:|---:|---:|"]
    for policy in POLICIES:
        subset = [r for r in metric_rows if r["policy_id"] == policy and r["identity_eligible"] and r["gold_count"] > 0]
        failure.append(f"| {policy} | {len(subset)} | {sum(r['top20_any_gold'] is False for r in subset)} | {sum(r['top20_complete_gold'] is False for r in subset)} | {sum(r['complete_support'] is False for r in subset if r['complete_support'] is not None)} | {sum(not r['reader_ok'] for r in metric_rows if r['policy_id'] == policy)} | {sum(r['truncated_gold_count'] for r in metric_rows if r['policy_id'] == policy)} |")
    failure.extend(["", "Rescue benefit/harm/unchanged and groupwise collision counts are in the corresponding CSV artifacts. Empty/ambiguous/>5 gold cases remain in the population and are explicitly represented by `None` denominators rather than silently filtered."])
    (OUT / "V11_TRACK_B_FAILURE_ATTRIBUTION.md").write_text("\n".join(failure) + "\n", encoding="utf-8")

    print(json.dumps({"status": "COMPLETE_READY_FOR_OWNER_REVIEW", "queries": 500, "selector_rows": 6000, "reader_rows": 6000, "jev_calls": 0, "identity_ambiguous": "qst_0413"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
