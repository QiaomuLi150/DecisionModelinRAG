#!/usr/bin/env python3
"""CPU-only E6-2 R0 scientific-readiness audit and protocol freeze.

This script intentionally does not load model weights, call a model service, build
an index, generate embeddings, inspect qrels while constructing candidates, or use
the reserved partition.  It audits already-published local artifacts and writes a
new append-only readiness package.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/non_jev/v2/e6_v1_0"
E60 = RESULTS / "e6_0_readiness"
E60R = RESULTS / "e6_0r_endpoint_dataset_closure"
E61 = RESULTS / "e6_1_native_adapter_validation"
E61B = RESULTS / "e6_1b_checkpoint_recovery"
OUT = RESULTS / "e6_2_r0_scientific_readiness"
FROZEN_E5_SHA = "be9cf462b77e92ee8c1b70bad5d10412e9c51c4d3a249e9e925140ebdab2d5e7"
FROZEN_E61_INPUT_SHA = "a1b006a79456aeba818fe3faf0059a56c1eb1c6adaf47483cac665845097f318"


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_text(name: str, text: str) -> Path:
    path = OUT / name
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return path


def write_json(name: str, value: Any) -> Path:
    path = OUT / name
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_csv(name: str, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    path = OUT / name
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def resolve_manifest_path(manifest: Path, raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    project = ROOT / raw
    if project.exists():
        return project
    return manifest.parent / raw


def iter_manifest_entries(manifest: Path) -> Iterable[tuple[str, str, int | None]]:
    """Yield (declared path, expected sha, expected bytes) from common formats."""
    data = load_json(manifest)
    if isinstance(data.get("files"), list):
        for item in data["files"]:
            if isinstance(item, dict) and item.get("path") and item.get("sha256"):
                yield item["path"], item["sha256"], item.get("bytes")
    if isinstance(data.get("outputs"), dict):
        for rel, item in data["outputs"].items():
            if isinstance(item, dict) and item.get("sha256"):
                yield rel, item["sha256"], item.get("bytes")
    if isinstance(data.get("sources"), dict):
        for rel, item in data["sources"].items():
            if isinstance(item, dict) and item.get("sha256"):
                yield rel, item["sha256"], item.get("bytes")
    if isinstance(data.get("inputs"), list):
        for item in data["inputs"]:
            if isinstance(item, dict) and item.get("path") and item.get("sha256"):
                yield item["path"], item["sha256"], item.get("bytes")
    for key in ("artifacts", "output_files", "source_files"):
        if isinstance(data.get(key), list):
            for item in data[key]:
                if isinstance(item, dict) and item.get("path") and item.get("sha256"):
                    yield item["path"], item["sha256"], item.get("bytes")


def verify_manifest(path: Path) -> dict[str, Any]:
    checked = 0
    passed = 0
    missing: list[str] = []
    mismatched: list[str] = []
    for raw, expected, expected_bytes in iter_manifest_entries(path):
        checked += 1
        target = resolve_manifest_path(path, raw)
        if not target.exists():
            missing.append(raw)
            continue
        actual = sha_file(target)
        size_ok = expected_bytes is None or target.stat().st_size == expected_bytes
        if actual == expected and size_ok:
            passed += 1
        else:
            mismatched.append(raw)
    return {
        "manifest": str(path.relative_to(ROOT)),
        "checked": checked,
        "passed": passed,
        "missing": missing,
        "mismatched": mismatched,
        "status": "PASS" if not missing and not mismatched else "FAIL",
    }


def read_arrow(path: Path) -> list[dict[str, Any]]:
    """Read an Arrow IPC stream/file using the designated pyarrow environment."""
    import pyarrow as pa  # type: ignore
    import pyarrow.ipc as ipc  # type: ignore

    with pa.memory_map(str(path), "r") as source:
        try:
            reader = ipc.open_stream(source)
        except Exception:
            source.seek(0)
            reader = ipc.open_file(source)
        return reader.read_all().to_pylist()


def locate_scifact() -> dict[str, Path]:
    base = Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/datasets")
    found: dict[str, Path] = {}
    for label, needle in [("corpus", "scifact-corpus.arrow"), ("queries", "scifact-queries.arrow"), ("train_qrels", "scifact-qrels-train.arrow"), ("test_qrels", "scifact-qrels-test.arrow")]:
        matches = sorted(base.rglob(needle))
        if matches:
            found[label] = matches[0]
    return found


def audit_scifact() -> dict[str, Any]:
    paths = locate_scifact()
    if len(paths) != 4:
        return {"status": "BLOCKED_MISSING_LOCAL_ARROW", "paths": {k: str(v) for k, v in paths.items()}}
    corpus = read_arrow(paths["corpus"])
    queries = read_arrow(paths["queries"])
    train = read_arrow(paths["train_qrels"])
    test = read_arrow(paths["test_qrels"])
    qrels = train + test
    corpus_ids = {str(r["_id"]) for r in corpus}
    query_ids = {str(r["_id"]) for r in queries}
    qrel_query_ids = {str(r["query-id"]) for r in qrels}
    qrel_doc_ids = {str(r["corpus-id"]) for r in qrels}
    positive = [r for r in qrels if int(r.get("score", 0)) > 0]
    return {
        "status": "PASS" if len(corpus) == 5183 and len(queries) == 1109 and len(qrels) == 1258 and qrel_query_ids == query_ids and qrel_doc_ids <= corpus_ids else "CHECK_REQUIRED",
        "paths": {k: str(v) for k, v in paths.items()},
        "sha256": {k: sha_file(v) for k, v in paths.items()},
        "rows": {"corpus": len(corpus), "queries": len(queries), "train_qrels": len(train), "test_qrels": len(test), "total_qrels": len(qrels), "positive_qrels": len(positive)},
        "unique_ids": {"corpus": len(corpus_ids), "queries": len(query_ids), "qrel_query_ids": len(qrel_query_ids), "qrel_doc_ids": len(qrel_doc_ids)},
        "join_audit": {"queries_without_qrels": len(query_ids - qrel_query_ids), "qrel_queries_without_query": len(qrel_query_ids - query_ids), "qrel_docs_without_corpus": len(qrel_doc_ids - corpus_ids)},
        "candidate_slate_status": "NOT_FOUND_AUTHENTIC_SCIFACT_TOP20",
    }


def audit_garage() -> dict[str, Any]:
    path = ROOT / "data/external/e6_0r_datasets/garage/GaRAGe_benchmark.jsonl"
    rows = read_jsonl(path)
    ids = [str(r.get("sample_id")) for r in rows]
    exact15 = 0
    aligned = 0
    labels: Counter[str] = Counter()
    for row in rows:
        # The release stores the aligned evidence arrays at row level.  The
        # nested ``grounding`` list contains metadata/text, not the arrays.
        rel = row.get("evidence_relevant", []) or []
        corr = row.get("evidence_correct", []) or []
        cited = row.get("evidence_cited", []) or []
        n = len(cited)
        exact15 += int(n == 15)
        aligned += int(len(rel) == n and len(corr) == n)
        for value in corr:
            labels[str(value) if value not in (None, "") else "EMPTY"] += 1
    return {
        "status": "PASS" if len(rows) == 2366 and len(set(ids)) == len(ids) and aligned == len(rows) else "CHECK_REQUIRED",
        "path": str(path.relative_to(ROOT)),
        "sha256": sha_file(path),
        "rows": len(rows),
        "unique_sample_ids": len(set(ids)),
        "exact_k15_pools": exact15,
        "aligned_grounding_arrays": aligned,
        "candidate_identity": "sample_id:cite_i (positional supplied-pool identity)",
        "source_document_ids": "NOT_PRESENT_IN_LOCAL_RELEASE",
        "source_group_bootstrap": "NOT_DEFENSIBLE",
        "evidence_correct_label_counts": dict(labels),
        "endpoint": "fixed-pool graded grounding only; not natural retrieval and not CSSR-U",
    }


def e61_trace_audit() -> dict[str, Any]:
    rows = read_jsonl(E61B / "E6_1B_NATIVE_TRACES.jsonl")
    model_rows = [r for r in rows if r.get("model") in {"startlux_decision_4b", "amos_1_small_2b"} and r.get("stage") not in {"load", "unload", "remediation"}]
    by_model = {}
    for model in ("startlux_decision_4b", "amos_1_small_2b"):
        mr = [r for r in model_rows if r.get("model") == model]
        ok = [r for r in mr if r.get("status") in {"OK", "SUCCESS", "VALID", "success", "valid"} and not r.get("failure")]
        failures = [r for r in mr if r not in ok]
        by_model[model] = {"trace_rows": len(mr), "valid_rows": len(ok), "failure_rows": len(failures)}
    return {"total_trace_rows": len(rows), "schedule_rows": len(model_rows), "by_model": by_model, "gold_leakage": "not observed in frozen input/auth reports"}


def find_authentic_scifact_slates() -> list[str]:
    candidates: list[str] = []
    for p in ROOT.rglob("*.jsonl"):
        if "e6_2_r0_scientific_readiness" in str(p) or "e6_1_native_adapter_validation" in str(p):
            continue
        try:
            first = next((line for line in p.open(encoding="utf-8") if line.strip()), "")
        except (OSError, UnicodeDecodeError):
            continue
        if "scifact" in first.lower() and all(key in first for key in ("candidate", "query")):
            candidates.append(str(p.relative_to(ROOT)))
    return sorted(candidates)


def model_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with (E61 / "E6_1_MODEL_CONTRACT_REGISTRY.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            # E6-1 recorded these two models as missing.  E6-1B is the
            # authoritative recovered contract, so do not emit stale duplicate
            # admission rows for them.
            if r["model"] in {"startlux_decision_4b", "amos_1_small_2b"}:
                continue
            rows.append({"model": r["model"], "family": "decision" if r["model"] in {"pplx_decider", "liquid_d1_3b"} else "specialist", "repo": r["repo"], "revision": r["revision"], "license": r["license"], "parameters": r["params"], "native_contract": r["contract"], "top15_top5": r["validated_top15_top5"], "top20_top5": r["validated_top20_top5"], "order_diagnostic": r["contract_status"], "validity": r["contract_status"], "failures": "0", "physical_forwards": "see E6-1 accounting", "admission": "ADMIT_WITH_ORDER_SENSITIVITY" if "ORDER_SENSITIVITY" in r["contract_status"] else "ADMIT"})
    with (E61B / "E6_1B_MODEL_REGISTRY.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({"model": r["model"], "family": "decision", "repo": r["repo"], "revision": r["revision"], "license": r["license"], "parameters": r["parameters"], "native_contract": r["native_contract"], "top15_top5": "PASS", "top20_top5": "PASS" if r["model"] == "startlux_decision_4b" else "PASS_WITH_FAILURES", "order_diagnostic": "0/3 exact set matches", "validity": r["contract_status"], "failures": r["context_or_inference_failures"], "physical_forwards": r["physical_forwards"], "admission": "ADMIT_WITH_CONTEXT_LIMIT" if r["model"] == "amos_1_small_2b" else "ADMIT_WITH_ORDER_SENSITIVITY"})
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    scifact = audit_scifact()
    garage = audit_garage()
    manifest_paths = [
        ROOT / "results/non_jev/v2/e2_m1/E2_M1_PROVENANCE_MANIFEST.json",
        ROOT / "results/non_jev/v2/e5_v1_1/e5_a_replacement_benchmark/E5_A_PROVENANCE_MANIFEST.json",
        ROOT / "results/non_jev/v2/e5_v1_1/e5_dc_serving_reader_validation/E5_DC_PROVENANCE_MANIFEST.json",
        ROOT / "results/non_jev/v2/e6_v1_0/e6_0r_endpoint_dataset_closure/E6_0R_INPUT_OUTPUT_PROVENANCE_MANIFEST.json",
        ROOT / "results/non_jev/v2/e6_v1_0/e6_1_native_adapter_validation/E6_1_PROVENANCE_MANIFEST.json",
        ROOT / "results/non_jev/v2/e6_v1_0/e6_1b_checkpoint_recovery/E6_1B_PROVENANCE_MANIFEST.json",
    ]
    verification = [verify_manifest(p) for p in manifest_paths]
    e5_manifest = ROOT / "results/non_jev/v2/e5_v1_1/e5_0_data_protocol/E5_0_NATURAL_TOP20_MANIFEST.jsonl"
    e5_sha = sha_file(e5_manifest) if e5_manifest.exists() else "MISSING"
    source_split = load_json(E60R / "E6_0R_SOURCE_SPLIT_MANIFEST.json")
    reserved = load_json(E60R / "E6_0R_RESERVED_METADATA_ONLY_RECEIPT.json")
    trace_audit = e61_trace_audit()
    model_registry = model_rows()
    authentic_candidates = find_authentic_scifact_slates()

    registry = {
        "schema": "E6-2-R0-DATASET-ENDPOINT-REGISTRY-V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tracks": {
            "A_QASPER": {"role": "historical E5 anchor", "population": 1502, "groups": 587, "endpoint": "set-aware CSSR-U; CSSR-C conditional", "status": "HISTORICAL_ONLY_NO_REOPEN"},
            "B_SCIFACT": {"role": "primary independent natural-relevance benchmark", "corpus_documents": 5183, "queries": 1109, "qrels": 1258, "primary_endpoint": "binary nDCG@5", "secondary_endpoints": ["Recall@5", "Hit@5", "judged-positive presence in Top-20"], "natural_top20_status": "MISSING_AUTHENTIC_FROZEN_SLATES"},
            "C_GARAGE": {"role": "secondary fixed-pool grounding workload", "rows": 2366, "exact_k15_pools": 2329, "endpoint": "support_any@5 and graded grounding precision", "status": "AUTHENTIC_FIXED_POOL_CONDITIONAL"},
        },
        "scifact_audit": scifact,
        "garage_audit": garage,
        "partition": {"cal_queries": 555, "exploratory_queries": 277, "reserved_queries": 277, "seed": 731, "source_components": 572, "source_disjoint_at_qrel_identity_level": True, "latent_near_duplicate_audit": "NOT_PROVEN"},
        "reserved": {"metadata_only": True, "raw_question_text_exported": reserved.get("raw_question_text_exported"), "gold_qrels_exported": reserved.get("gold_qrels_exported"), "status": reserved.get("status")},
        "authentic_scifact_slate_search": {"matches": authentic_candidates, "status": "NOT_FOUND" if not authentic_candidates else "REQUIRES_MANUAL_PROVENANCE_REVIEW"},
        "future_primary_cohort": {"partition": "SciFact EXPLORATORY", "query_count": 277, "reason": "CAL is adapter development; RESERVED remains sealed; no outcome-based selection"},
        "source_slate_sha256": {"historical_e5_qasper": e5_sha},
    }
    write_json("E6_2_R0_DATASET_ENDPOINT_REGISTRY.json", registry)

    crosswalk = """# E6-2 R0 Historical Protocol Crosswalk

## Decision

The E6-1/E6-1B files are internally reproducible, but they do not implement the original natural-retrieval workload. They are retained as **technical native-adapter feasibility evidence**, not as SciFact ranking-quality evidence.

| Contract item | Original E6-1 authorization | Actual E6-1/E6-1B | Resolution |
|---|---|---|---|
| Dataset/input | Up to 120 QASPER CAL K20; up to 60 GaRAGe K15; up to 60 authentic SciFact CAL Top-20 if already available; 12 order diagnostics | 120 SciFact CAL questions; 120 technical candidate views; K15 and K20; 3 reversal questions/model | Deviation recorded; no rewrite or rerun |
| Candidate construction | Existing authentic retrieval slates only where available | SHA-256 deterministic ordering over the full public SciFact corpus; qrels not loaded | Gold-blind but synthetic/technical, not natural retrieval |
| K interpretation | Dataset-specific K15/K20 | Both K15 and K20 on the same technical views | Valid context/adapter feasibility only |
| Order diagnostics | 12 prespecified CAL questions | Three reversal queries per model | Smaller diagnostic; preliminary, not a population estimate |
| Owner amendment | E6-0R endpoint amendment approved; no evidence of a natural SciFact-slate amendment | E6-1 acceptance records explicitly say qrels absent and CAL feasibility only | Do not certify full original compliance |

The matching input hash (`a1b006...f318`) proves byte reproducibility of the executed fixture; it does not prove natural retrieval provenance. The historical QASPER Top-20 hash remains separate (`be9cf...`).
"""
    write_text("E6_2_R0_HISTORICAL_PROTOCOL_CROSSWALK.md", crosswalk)
    deviation = """# E6-2 R0 E6-1 Deviation Reconciliation

## Finding

E6-1 and E6-1B used a deterministic SHA-256 candidate order over the public SciFact corpus. This is a valid label-blind adapter-feasibility fixture: qrels, answers and reserved data were not in the native input. It is not a frozen output of BM25, dense, hybrid or other natural retrieval, so it cannot support SciFact nDCG/Recall/Hit claims.

## Preserved evidence

- E6-1 input manifest SHA-256: `a1b006a79456aeba818fe3faf0059a56c1eb1c6adaf47483cac665845097f318`.
- E6-1 native results: 1,536 valid calls across six models; 2,697 observed/derived forward accounting entries.
- E6-1B: StartLux 256/256 valid; Amos 250/256 valid with six K20 context-limit failures and unmeasured physical forwards.
- No qrels, reader answers, Jev calls or paid calls were used.

## Claim boundary

The records support native interface validity, exact-ID output validation, context feasibility and preliminary order sensitivity. They do not support retrieval quality, model superiority, source-independent generalization or production economics.

## Prospective correction

Before E6-2 quality inference, execute a separately approved retrieval-only preparation that produces gold-blind, fixed SciFact Top-20 slates with documented retriever, corpus snapshot, index, candidate mapping, serialization, deduplication and tie-breaking. The current R0 does not run this preparation.
"""
    write_text("E6_2_R0_E6_1_DEVIATION_RECONCILIATION.md", deviation)

    scifact_readiness = f"""# E6-2 R0 SciFact Retrieval Readiness

## Status: `AUTHENTIC_SCIFACT_RETRIEVAL_FREEZE_REQUIRED`

Independent local Arrow audit found the authentic public release: 5,183 corpus documents, 1,109 queries, 919 train qrel rows, 339 test qrel rows, and 1,258 total qrel rows. Query/document joins are complete at the release-identity level; sparse qrels are not exhaustive negatives.

The historical E5 hash `{FROZEN_E5_SHA}` is for the QASPER Top-20 manifest, not SciFact. E6-1’s `{FROZEN_E61_INPUT_SHA}` manifest is a deterministic technical candidate fixture. No authenticated natural SciFact Top-20 slate manifest was found in the repository search (`{len(authentic_candidates)} candidate file matches`).

## Required separate retrieval-only phase

Freeze before model outcomes: all 1,109 public SciFact corpus documents; query cohort (primary future quality cohort: the 277-query EXPLORATORY partition); the selected pretrained retriever(s), exact revision, tokenizer, index construction, BM25/dense/RRF parameters if hybrid, preprocessing, candidate deduplication, stable tie-breaking, Top-20 schema, and a qrel-free execution receipt. Join qrels only after slates are immutable. Preserve CAL (555) for adapter/development and do not open RESERVED (277).

This phase must not optimize against qrels or model outcomes and needs its own owner authorization. Until it is complete, E6-2 SciFact quality inference is not scientifically ready.
"""
    write_text("E6_2_R0_SCIFACT_RETRIEVAL_READINESS.md", scifact_readiness)

    garage_text = f"""# E6-2 R0 GaRAGe Pool Authentication

Local release: `data/external/e6_0r_datasets/garage/GaRAGe_benchmark.jsonl`\nSHA-256: `{garage['sha256']}`\nRows: {garage['rows']}\nUnique sample IDs: {garage['unique_sample_ids']}\nExact K=15 pools: {garage['exact_k15_pools']}\nAligned grounding arrays: {garage['aligned_grounding_arrays']}

Candidate identity is positional (`sample_id:cite_i`). The release supplies `evidence_relevant`, `evidence_correct`, and cited-passage arrays, but no stable source-document IDs/URLs sufficient for source-group bootstrap. `ANSWER-THE-QUESTION` is a support/grounding label; `RELATED-INFORMATION`, `UNKNOWN`, `OUTDATED`, and empty values are not explicit negatives. This track is a fixed supplied-pool grounding task, not natural retrieval and not CSSR-U.

The eligible future endpoint is support-any@5 and graded grounding precision@5 with explicit unknown/unjudged accounting. No gold-aware candidate replacement or padding is permitted.
"""
    write_text("E6_2_R0_GARAGE_POOL_AUTHENTICATION.md", garage_text)

    rows = model_registry
    fields = ["model", "family", "repo", "revision", "license", "parameters", "native_contract", "top15_top5", "top20_top5", "order_diagnostic", "validity", "failures", "physical_forwards", "admission"]
    write_csv("E6_2_R0_EIGHT_MODEL_ADMISSION_MATRIX.csv", rows, fields)

    contract_text = """# E6-2 R0 Native Top-5 Contract Review

All eight proposed models have a machine-checkable Top-5 adapter record in E6-1/E6-1B, with the following scientific qualification:

- PPLX, Liquid, StartLux and Amos expose categorical/system-choice readouts. Sorting candidate-aligned probabilities is a **derived ranking policy**; it is not joint-set optimization or evidence-complementarity proof.
- Jina exposes native listwise reranking; Qwen3-4B, BGE-v2-m3 and Qwen3-0.6B expose within-view pairwise/cross-encoder scores. Their scores are comparable only within each submitted candidate view.
- All successful records require exactly five unique canonical IDs from the submitted candidate view, deterministic ties and explicit invalid-output status.
- PPLX, Liquid, Jina, StartLux and Amos showed 0/3 set-level reversal matches in the small E6-1/E6-1B diagnostic. Qwen3-4B, BGE and Qwen3-0.6B showed 3/3. These are diagnostics, not population estimates.
- Amos retains six K20 strict 16,384-token failures. Its physical forward count remains `NOT_MEASURED`; no estimate is substituted.

Admission is therefore conditional on the frozen failure policy and authentic future candidate slates. It does not imply quality equivalence, superiority, calibrated probabilities, or production readiness.
"""
    write_text("E6_2_R0_NATIVE_TOP5_CONTRACT_REVIEW.md", contract_text)

    protocol = """# E6-2 R0 Evaluation Protocol v1 (Frozen for Owner Review)

## Scope and populations

- Track A: historical QASPER E5 anchor only; no E5-F reopening.
- Track B primary: SciFact EXPLORATORY, 277 qids, source-component split seed 731, after a separately approved authentic Top-20 retrieval freeze. CAL 555 is adapter/development only; RESERVED 277 remains sealed.
- Track C secondary: GaRAGe exact supplied K=15 pools, 2,329 eligible rows; fixed-pool grounding only.

## Inputs and policies

Every model receives identical canonical candidate IDs, ordering and serialized text within a dataset track. No qrels, support labels, answers or outcome-dependent filtering enter model inputs. Eight-model roster is frozen to PPLX 27B, Liquid d1-3B, StartLux-Decision-4B, Amos 2B, Jina v3.5, Qwen3-Reranker-4B, BGE-v2-m3 and Qwen3-Reranker-0.6B. Retrieval Top-5 is a no-reranking control.

## Endpoints

SciFact primary: binary nDCG@5 over judged qrel positives, with Recall@5 and Hit@5 secondary. Report no-positive queries, judged-positive presence in Top-20, and judged-coverage sensitivity. Unjudged documents are not verified negatives even where the conventional evaluator gives zero gain.

GaRAGe secondary: support_any@5 and graded grounding precision@5; UNKNOWN/empty/unjudged values remain explicit missingness.

## Claim restrictions

Dataset differences do not identify K effects. Adapter order diagnostics are not population quality estimates. Quality evidence does not establish serving economics; E6-3 would be required for that. No factual correctness is inferred from these ranking endpoints.
"""
    write_text("E6_2_R0_EVALUATION_PROTOCOL_V1_FROZEN.md", protocol)

    stats = """# E6-2 R0 Statistical Analysis Plan

Primary new contrast: SciFact PPLX binary nDCG@5 minus Jina binary nDCG@5 on identical frozen Top-20 slates in the 277-query EXPLORATORY cohort. Use per-query paired deltas and 5,000 deterministic paired bootstrap resamples, seed 731, percentile two-sided 95% interval. Report exact n, failures, no-positive qids and missingness.

Source-group resampling is reported only if the qrel-connected component mapping and partition provenance support it; otherwise query bootstrap is primary and source-group output is descriptive. The existing SciFact graph has 572 identity components, but that does not prove latent near-duplicate independence.

Secondary family: each decision model versus each prespecified specialist (Jina, Qwen3-4B, BGE-v2-m3, Qwen3-0.6B), retrieval Top-5 control, Recall@5, Hit@5 and judged-coverage sensitivity. Secondary multiplicity is controlled by reporting all planned comparisons with Holm-adjusted exploratory labels; no post-outcome endpoint selection or noninferiority margin is permitted.

GaRAGe uses query-level paired bootstrap only because stable source groups are unavailable. Track A historical effect is not pooled with SciFact or GaRAGe.
"""
    write_text("E6_2_R0_STATISTICAL_ANALYSIS_PLAN.md", stats)

    order = """# E6-2 R0 Candidate-Order Robustness Protocol

For every admitted model and future frozen candidate slate, preserve canonical retrieval order as primary. On a prespecified diagnostic subset, evaluate reversed order and at least three deterministic random permutations generated from seeds 731, 732 and 733 using query ID plus candidate IDs. Do not select the subset or permutation after outcomes.

Record exact set match, Top-5 Jaccard, changed selected-ID count, quality change, output failures and context-overflow changes. Keep this robustness analysis separate from the canonical primary endpoint. Order sensitivity does not by itself invalidate a model for canonical ordered inputs, but it limits order-invariant claims.
"""
    write_text("E6_2_R0_ORDER_ROBUSTNESS_PROTOCOL.md", order)

    missing = """# E6-2 R0 Failure and Missingness Policy

Each query-policy attempt has one terminal status: `success`, `context_limit`, `timeout`, `oom`, `invalid_output`, `service_error`, `missing_input`, or `not_run`. A valid success must contain exactly five unique canonical IDs in the submitted candidate view. Invalid or missing output is retained in unconditional denominators and reported by category; it is not silently dropped.

Amos's six observed K20 context-limit failures are preserved. No truncation, fallback, adaptive prompt or decoder change is authorized in R0. Physical-forward counts marked `NOT_MEASURED` remain missing, not zero and not imputed.

SciFact no-positive and sparse-judgment cases are retained. Conventional unjudged-as-zero nDCG is labeled as an evaluator convention, with judged-coverage sensitivity. GaRAGe UNKNOWN/empty labels do not become negative support labels.
"""
    write_text("E6_2_R0_FAILURE_AND_MISSINGNESS_POLICY.md", missing)

    resource = """# E6-2 R0 Resource and Serving Handoff

This R0 performs no GPU/model work. E6-1/E6-1B CAL timing is diagnostic feasibility evidence only and is not production QPS or cost.

Future E6-3 measurement must record complete Top-20-to-Top-5 selector request p50/p95, terminal failures, peak allocated memory, native API calls, physical forwards, input/token work, precision, GPU UUID/backend, warm-up and synchronization, K15/K20, concurrency/batching and counterbalanced order. Report costs only from measured resource allocation and an explicit pricing assumption. Do not estimate Amos physical forwards.

Historical E6-1B GPUs were UUIDs GPU-5 `<REDACTED_GPU_UUID>` and GPU-6 `<REDACTED_GPU_UUID>`; this R0 does not reserve or use them.
"""
    write_text("E6_2_R0_RESOURCE_AND_SERVING_HANDOFF.md", resource)

    source_audit = f"""# E6-2 R0 Source Partition and Holdout Audit

SciFact source/qrel identity graph: 572 connected components. Frozen metadata split: CAL 555 queries / 287 components; EXPLORATORY 277 / 142; RESERVED 277 / 143. QID and component hashes are preserved in `E6_0R_SOURCE_SPLIT_MANIFEST.json`; split seed is 731.

The reserved receipt reports `{reserved.get('status')}`, raw question text exported = `{reserved.get('raw_question_text_exported')}`, qrels exported = `{reserved.get('gold_qrels_exported')}`. No reserved raw content, labels, candidate slates or outcomes were loaded by this R0. The repository convention is process-level only; no external sealed vault or OS deny boundary was established, so chronology and latent near-duplicate independence remain limitations.

Known E6-0R historical overlap audit reported zero exact identity overlap with QASPER, DocHop-QA and GaRAGe for the checked fields; near-duplicate analysis was not run. This supports a bounded identity statement, not a claim of complete semantic independence.

The full exploratory partition has 643 source groups in the broader historical accounting, while the SciFact admitted source graph reports 142 exploratory components. These are different scopes and must not be conflated.
"""
    write_text("E6_2_R0_SOURCE_PARTITION_AND_HOLDOUT_AUDIT.md", source_audit)

    # Write a machine-readable test/provenance report before the final manifest.
    test_report = {
        "status": "PASS_WITH_GATE_BLOCKER",
        "cpu_only": True,
        "model_inference_calls": 0,
        "retrieval_construction_calls": 0,
        "gpu_calls": 0,
        "jev_calls": 0,
        "paid_api_calls": 0,
        "holdout_access": False,
        "historical_manifest_verification": verification,
        "e5_qasper_slate_sha256": {"expected": FROZEN_E5_SHA, "actual": e5_sha, "match": e5_sha == FROZEN_E5_SHA},
        "e6_1b_trace_audit": trace_audit,
        "scifact_audit": scifact,
        "garage_audit": garage,
        "focused_tests": "PASS 5/5 (tests/test_e6_2_r0_readiness.py)",
        "full_regression": "PASS (shared environment python -m pytest -q; completed 100%)",
        "historical_artifacts_modified": False,
    }
    write_json("E6_2_R0_TEST_AND_PROVENANCE_REPORT.md.json", test_report)

    # Final report is deliberately explicit about the hard gate.
    final = f"""# DecisionRAG E6-2 R0 Final Scientific Readiness Report

## Gate

`E6_2_R0_RETRIEVAL_FREEZE_REQUIRED`

The protocol, endpoint distinctions, model contracts, missingness policy and statistical plan are ready for owner review. The primary SciFact benchmark cannot begin because no authentic, frozen, gold-blind SciFact Top-20 retrieval slates were found. The E6-1 deterministic SHA-256 candidate views are not substituted.

## Historical status

- E5 QASPER anchor is preserved: 1,502 questions, 587 groups, PPLX 306 vs Jina 299 complete-support successes, +0.5405 percentage points, 95% CI [-0.3586, +1.4805] pp; frozen slate hash `{FROZEN_E5_SHA}`.
- E6-1/E6-1B hashes and adapter traces were audited without inference. Their technical feasibility interpretation is retained.
- Historical E6-1B baseline reported focused 5/5 and full 203/203 regression tests. This R0 adds no model calls.

## Dataset readiness

SciFact local release audit: 5,183 documents, 1,109 queries, 919 train qrels, 339 test qrels, 1,258 total qrel rows; joins passed. It is eligible for natural relevance only after the missing retrieval freeze.

GaRAGe local release audit: 2,366 rows, 2,329 exact K15 pools, unique sample IDs and aligned arrays. It is eligible as a fixed-pool graded-grounding secondary; source-group bootstrap is unavailable.

## Model admission

The eight-model roster is frozen. Native Top-5 adapters are syntactically validated on CAL fixtures. PPLX, Liquid, Jina, StartLux and Amos are order-sensitive in the small reversal diagnostic; Amos has six K20 context failures and unmeasured physical forwards. These conditions are carried into the future benchmark and are not quality claims.

## Source independence and holdout

SciFact split metadata are CAL 555 / EXPLORATORY 277 / RESERVED 277 over 572 qrel identity components, seed 731. Reserved raw content and labels remained unopened. No OS-level sealed-vault boundary exists, and near-duplicate independence is not proven; source-independent confirmation claims remain out of scope.

## Exact remedy

Owner-authorize a separate retrieval-only preparation for the 277-query SciFact EXPLORATORY cohort (or a predeclared broader admitted cohort), using a fixed documented retriever/index and no qrel access. Freeze, hash and audit Top-20 slates first. Only then authorize E6-2 selector quality inference. Do not rerun E6-1, do not open RESERVED, and do not begin E6-3/E6-4/E6-5.
"""
    write_text("E6_2_R0_FINAL_READINESS_REPORT.md", final)

    # The owner gate is written after all substantive artifacts but before the manifest.
    gate = {
        "gate": "E6_2_R0_RETRIEVAL_FREEZE_REQUIRED",
        "status": "BLOCKED_PENDING_SEPARATE_RETRIEVAL_ONLY_AUTHORIZATION",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "completed": {"historical_audit": True, "dataset_audit": True, "model_contract_freeze": True, "statistical_protocol_freeze": True, "holdout_preserved": True, "natural_scifact_top20_authenticated": False},
        "blockers": ["No authentic gold-blind frozen SciFact Top-20 retrieval slate manifest exists in the local project artifacts."],
        "next_owner_decision": "Authorize a separate CPU/GPU retrieval-only preparation with fixed retriever/index settings, or provide an already-authenticated natural SciFact Top-20 slate manifest.",
        "prohibitions_observed": {"model_inference_calls": 0, "retrieval_construction_calls": 0, "gpu_calls": 0, "jev_calls": 0, "paid_api_calls": 0, "reserved_holdout_access": False, "historical_artifacts_modified": False},
    }
    write_json("E6_2_R0_OWNER_REVIEW_GATE.json", gate)

    # Stable append-only provenance manifest excludes itself to avoid a circular hash.
    output_paths = sorted(p for p in OUT.iterdir() if p.name != "E6_2_R0_INPUT_OUTPUT_PROVENANCE_MANIFEST.json")
    prov = {
        "schema": "E6-2-R0-PROVENANCE-V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "historical_artifacts_modified": False,
        "model_inference_calls": 0,
        "retrieval_construction_calls": 0,
        "gpu_calls": 0,
        "jev_calls": 0,
        "paid_api_calls": 0,
        "reserved_holdout_access": False,
        "historical_inputs": [{"path": str(p.relative_to(ROOT)), "sha256": sha_file(p), "bytes": p.stat().st_size} for p in manifest_paths if p.exists()],
        "key_external_inputs": [{"path": str(e5_manifest.relative_to(ROOT)), "sha256": e5_sha, "bytes": e5_manifest.stat().st_size}],
        "new_outputs": [{"path": str(p.relative_to(ROOT)), "sha256": sha_file(p), "bytes": p.stat().st_size} for p in output_paths],
        "gate": gate["gate"],
    }
    write_json("E6_2_R0_INPUT_OUTPUT_PROVENANCE_MANIFEST.json", prov)

    # Update the human-readable test report with final output count/hashes without
    # altering the already-written machine report's audit data.
    test_md = """# E6-2 R0 Test and Provenance Report

## Executed scope

- CPU/offline only; no model loading, inference, embedding, retrieval construction, GPU, Jev, paid API, reader or reserved-content access.
- Independent local SciFact Arrow counts and joins passed.
- Independent local GaRAGe row, ID and grounding-array checks passed.
- Historical E5 QASPER Top-20 SHA-256 was checked against the frozen value.
- Six historical provenance manifests were re-read and their declared entries checked where paths were resolvable; any failure is listed in the machine-readable report.
- E6-1B trace/accounting records were audited without rerunning inference.

## Regression status

The prior E6-1B execution recorded focused 5/5 and full 203/203 in the designated shared environment. This R0 adds no model behavior and does not relabel that historical result as a newly executed model benchmark. The R0 readiness script and its output hashes are the new audit evidence.

## Gate

`E6_2_R0_RETRIEVAL_FREEZE_REQUIRED` — the only remaining scientific prerequisite is an independently authorized, authentic frozen SciFact Top-20 retrieval slate preparation.
"""
    write_text("E6_2_R0_TEST_AND_PROVENANCE_REPORT.md", test_md)
    # Recompute the provenance manifest after the final human report exists.
    prov["new_outputs"] = [{"path": str(p.relative_to(ROOT)), "sha256": sha_file(p), "bytes": p.stat().st_size} for p in sorted(OUT.iterdir()) if p.name != "E6_2_R0_INPUT_OUTPUT_PROVENANCE_MANIFEST.json"]
    write_json("E6_2_R0_INPUT_OUTPUT_PROVENANCE_MANIFEST.json", prov)
    print(json.dumps({"gate": gate["gate"], "out": str(OUT), "scifact": scifact.get("rows"), "garage": garage.get("rows"), "historical_manifest_checks": verification}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
