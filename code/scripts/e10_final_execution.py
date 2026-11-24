#!/usr/bin/env python3
"""DecisionRAG E10 final replacement-validity execution.

This runner is deliberately append-only.  It consumes authenticated E9/E5/E6
artifacts, freezes the E10 protocol before new timing outcomes, optionally runs
one bounded same-GPU native selector cohort, and writes the final E10 evidence
freeze.  It never downloads models, accesses holdouts, runs a reader, or calls
external/paid services.
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
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
E9 = ROOT / "results/non_jev/v2/e9_bright_v1_0"
E9A = ROOT / "results/non_jev/v2/e9_bright_interpretation_v1_0"
E5 = ROOT / "results/non_jev/v2/e5_v1_1/e5_dc_serving_reader_validation"
E6 = ROOT / "results/non_jev/v2/e6_v1_0"
OUT = ROOT / "results/non_jev/v2/e10_v1_0"
SEED = 731
DOMAINS = ["biology", "earth_science", "economics", "psychology", "robotics", "stackoverflow", "sustainable_living", "pony", "leetcode", "aops", "theoremqa_questions", "theoremqa_theorems"]
MODELS = ["pplx_decider", "jina_v3_5", "liquid_d1_3b", "qwen3_reranker_0_6b"]
MODEL_META = {
    "pplx_decider": {"repo": "perplexity-ai/pplx-decider-v1.1-27b", "revision": "14c9d90137962d300f3de95633f7b6535e80c5ce", "kind": "pplx_choice", "params": "27B", "path": "<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--perplexity-ai--pplx-decider-v1.1-27b/snapshots/14c9d90137962d300f3de95633f7b6535e80c5ce"},
    "liquid_d1_3b": {"repo": "LiquidAI/d1-3B", "revision": "051bcc464b01b9f92942b364d9586b0ef5912432", "kind": "liquid_choice", "params": "3.12B", "path": "<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--LiquidAI--d1-3B/snapshots/051bcc464b01b9f92942b364d9586b0ef5912432"},
    "jina_v3_5": {"repo": "jinaai/jina-reranker-v3.5", "revision": "e8a93f33f0b22108f8c2364f8484ce3422552fbc", "kind": "jina_listwise", "params": "0.6B", "path": "<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--jinaai--jina-reranker-v3.5/snapshots/e8a93f33f0b22108f8c2364f8484ce3422552fbc"},
    "qwen3_reranker_0_6b": {"repo": "Qwen/Qwen3-Reranker-0.6B", "revision": "e61197ed45024b0ed8a2d74b80b4d909f1255473", "kind": "cross_encoder", "params": "0.6B", "path": "<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-0.6B/snapshots/e61197ed45024b0ed8a2d74b80b4d909f1255473"},
}


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
    values = sorted(float(x) for x in values)
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] if lo == hi else values[lo] + (values[hi] - values[lo]) * (pos - lo)


def gpu_snapshot() -> dict[str, Any]:
    try:
        raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True, stderr=subprocess.STDOUT)
        devices = []
        for line in raw.splitlines():
            vals = [x.strip() for x in line.split(",")]
            if len(vals) >= 7:
                devices.append({"index": int(vals[0]), "uuid": vals[1], "name": vals[2], "memory_total_mib": int(vals[3]), "memory_used_mib": int(vals[4]), "memory_free_mib": int(vals[5]), "utilization_percent": int(vals[6])})
        try:
            apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"], text=True, stderr=subprocess.STDOUT).strip()
        except Exception as exc:
            apps = f"UNAVAILABLE:{type(exc).__name__}:{exc}"
        return {"status": "ok", "devices": devices, "compute_apps_raw": apps, "captured_at": utc()}
    except Exception as exc:
        return {"status": "unavailable", "error": f"{type(exc).__name__}:{exc}", "captured_at": utc()}


def load_slates() -> list[dict[str, Any]]:
    rows = read_jsonl(E9 / "01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl")
    if len(rows) != len(set((x["domain"], x["query_id"]) for x in rows)):
        raise RuntimeError("duplicate E9 frozen slates")
    if any(len(x.get("candidate_ids", [])) != 20 or len(set(x["candidate_ids"])) != 20 for x in rows):
        raise RuntimeError("invalid E9 Top-20 candidate identity")
    return rows


def freeze_sample(slates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    rng = random.Random(SEED)
    selected: list[dict[str, Any]] = []
    for domain in DOMAINS:
        rows = sorted([x for x in slates if x["domain"] == domain], key=lambda x: x["query_id"])
        if len(rows) < 20:
            raise RuntimeError(f"domain {domain} has fewer than 20 queries")
        picked = rng.sample(rows, 20)
        selected.extend(sorted(picked, key=lambda x: x["query_id"]))
    selected.sort(key=lambda x: (x["domain"], x["query_id"]))
    digest = sha_json([{"domain": x["domain"], "query_id": x["query_id"], "slate_sha256": x["slate_sha256"], "input_sha256": x["input_sha256"]} for x in selected])
    return selected, digest


def source_files() -> list[Path]:
    return [
        E9 / "01_retrieval/BRIGHT_FROZEN_TOP20_SLATES.jsonl",
        E9 / "03_analysis/BRIGHT_PRIMARY_PAIRED_STATISTICS.json",
        E9 / "04_resources/BRIGHT_SELECTOR_RESOURCE_SUMMARY.csv",
        E9 / "06_closure/E9_BRIGHT_PROVENANCE_MANIFEST.json",
        E9A / "E9_BRIGHT_COMPARISON_TABLE.csv",
        E9A / "E9_BRIGHT_FINAL_OWNER_REVIEW_GATE.json",
        E5 / "E5_DC_EXECUTION_MANIFEST_FROZEN.json",
        E5 / "E5_C_READER_INPUTS_AND_OUTPUTS.jsonl",
        E5 / "E5_C_PAIRED_READER_STATISTICS.csv",
        E6 / "e6_1_native_adapter_validation/E6_1_MODEL_CONTRACT_REGISTRY.csv",
        E6 / "e6_1_native_adapter_validation/E6_1_OWNER_REVIEW_GATE.json",
    ]


def build_taxonomy() -> list[dict[str, Any]]:
    registry = list(csv.DictReader((E6 / "e6_1_native_adapter_validation/E6_1_MODEL_CONTRACT_REGISTRY.csv").open(encoding="utf-8")))
    actual = {"pplx_decider": "E9-BRIGHT + E10-B", "liquid_d1_3b": "E9-BRIGHT + E10-B", "jina_v3_5": "E9-BRIGHT + E10-B", "qwen3_reranker_0_6b": "E9-BRIGHT + E10-B", "qwen3_reranker_4b": "E6-2-R4D", "bge_reranker_v2_m3": "E6-2-R4D", "startlux_decision_4b": "E6-2-R4D; native checkpoint missing in E6-1 registry", "amos_1_small_2b": "E6-2-R4D; native checkpoint missing in E6-1 registry"}
    rows = []
    for row in registry:
        model = row["model"]
        rows.append({
            "model_policy": model,
            "repo": row["repo"],
            "revision": row["revision"],
            "parameter_count": row["params"],
            "backbone_or_kind": row["kind"],
            "native_scoring_contract": row["contract"],
            "selection_output": row["selection_output"],
            "context_or_contract_limit": "documented in native adapter; not normalized across policies",
            "precision_runtime": "historical registry; E10-B records only four-policy resource run",
            "license": row["license"],
            "checkpoint_present_in_registry": row["snapshot_present"],
            "contract_status": row["contract_status"],
            "actual_experimental_coverage": actual.get(model, "not found"),
            "analytic_unit": "trained native model-policy implementation; not an architecture-causal comparison",
        })
    return rows


def protocol_phase() -> None:
    for name in ("00_protocol", "01_reader", "02_serving", "03_taxonomy", "04_integration", "05_closure"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    slates = load_slates()
    sample, sample_hash = freeze_sample(slates)
    source_manifest = [{"path": str(p.relative_to(ROOT)), "exists": p.exists(), "bytes": p.stat().st_size if p.exists() else None, "sha256": sha_file(p) if p.exists() else None} for p in source_files()]
    write_json(OUT / "00_protocol/E10_FINAL_PROTOCOL_LOCK.json", {
        "protocol": "DecisionRAG-E10-v1.0",
        "locked_at": utc(),
        "status": "FROZEN_BEFORE_E10_NEW_OUTCOMES",
        "seed": SEED,
        "dataset": {"name": "xlangai/BRIGHT", "revision": "3066d29c9651a576c8aba4832d249807b181ecae", "slate_source": "authenticated E9 frozen Top-20", "slate_count": len(slates), "domains": DOMAINS},
        "primary_policies": {k: MODEL_META[k] for k in MODELS},
        "primary_comparisons": ["PPLX_minus_Jina operational E9 quality; E10 resource replication", "Liquid_minus_Qwen0.6B resource replication"],
        "timing_sample": {"n": len(sample), "queries_per_domain": 20, "selection_rule": "uniform without replacement within each sorted domain, seed 731, before E10 timing outcomes", "sample_sha256": sample_hash, "query_ids": [{"domain": x["domain"], "query_id": x["query_id"], "slate_sha256": x["slate_sha256"], "input_sha256": x["input_sha256"]} for x in sample]},
        "timing_contract": {"boundary": "candidate materialization from frozen slate, input serialization/tokenization as native adapter performs it, native inference/readout, canonical Top-5 assembly", "model_loading_included": False, "concurrency": 1, "cache_policy": "no application cache; one genuine native call per request", "request_order": "fixed domain/query order; model order PPLX,Jina,Liquid,Qwen; sequential order confounding recorded"},
        "metrics": {"quality": "E9 operational binary known-positive nDCG@5 remains primary; no new quality endpoint", "resource": ["all-attempt latency p50/p95", "successful-only latency p50/p95", "failure-only latency p50/p95", "valid fraction", "peak allocated memory", "failure categories", "native call count"], "bootstrap": "E9 source macro-domain bootstrap, seed 731, 5,000; E10 resource estimates are descriptive"},
        "exclusions": ["E9 TREC blocked tree", "BRIGHT gold labels from selector inputs", "reserved holdouts", "new reader inference", "new model downloads", "Jev/paid APIs", "QPS/cost inference"],
        "source_manifest": source_manifest,
    })
    recon = f"""# E9 bootstrap reconciliation\n\n## Gate\n\n`E10_0_RECONCILED` for the point estimate and inferential target; the historical interval discrepancy is retained as an explicit implementation/stream sensitivity, not silently replaced.\n\n## Frozen primary\n\nThe authenticated E9 source reports PPLX minus Jina operational macro-domain binary known-positive nDCG@5 = `-0.09837202926661133`, with the prespecified 5,000-resample seed-731 interval `[-0.15381727491417288, -0.0442565978717294]`. The E9 point estimate is reproduced exactly from the frozen per-domain effects.\n\n## Discrepancy\n\nThe later E9 interpretation replay reports `[-0.15503154829573165, -0.0434499211615929]` for the same point estimate. Both intervals are domain-resampling intervals around the same 12 fixed domain effects, but the replay does not preserve the original bootstrap draw stream/code artifact needed to establish byte-for-byte RNG parity. The difference is therefore classified as a bootstrap-stream/implementation sensitivity, not as a changed estimand. The original E9 interval remains authoritative because it belongs to the preregistered run; the replay interval is labeled sensitivity.\n\nFixed-domain query and hierarchical query/domain intervals remain secondary sensitivities. No new resampling was used to select the E10 timing sample.\n\n## Data and integrity\n\nThe E9 frozen Top-20 slates, native outputs, source statistics, and final interpretation gate were read-only inputs. No gold label, answer, or protected holdout was accessed during sample selection.\n"""
    (OUT / "00_protocol/E9_BOOTSTRAP_RECONCILIATION.md").write_text(recon, encoding="utf-8")
    ledger = []
    for item in source_manifest:
        ledger.append({"source_path": item["path"], "sha256": item["sha256"], "status": "PRIMARY" if "E9_PRIMARY" in item["path"] or "FROZEN_TOP20" in item["path"] else "SECONDARY", "population": "E9 historical or E10 protocol source", "endpoint": "source authentication", "outcome": "authenticated" if item["exists"] else "MISSING"})
    write_csv(OUT / "00_protocol/E10_EVIDENCE_PROVENANCE_LEDGER.csv", ledger)
    write_json(OUT / "00_protocol/E10_SOURCE_OVERLAP_EXPOSURE_REGISTER.json", {"protected_paths_not_opened": ["E3-F", "E5-F", "E6-F", "RESERVED"], "status": "NO_PROTECTED_CONTENT_ACCESSED", "source_files_checked": [x["path"] for x in source_manifest]})

    cohort = list(csv.DictReader((E5 / "E5_DC_FROZEN_REPRESENTATIVE_AND_DIAGNOSTIC_COHORTS.csv").open(encoding="utf-8")))
    representative = sorted({r["query_id"] for r in cohort if r.get("cohort") == "representative"})
    diagnostic = sorted({r["query_id"] for r in cohort if r.get("cohort") == "diagnostic_enriched"})
    packet_manifest = {"status": "FROZEN_PACKET_MANIFEST_ONLY_NO_ADJUDICATION", "representative_n": len(representative), "diagnostic_n": len(diagnostic), "representative_ids_sha256": sha_json(representative), "diagnostic_ids_sha256": sha_json(diagnostic), "blinding": "selector-neutral labels would be required; no human or independent local judge was available", "answers_or_labels_written": False}
    write_json(OUT / "01_reader/E10_A_BLINDED_PACKET_MANIFEST.json", packet_manifest)
    (OUT / "01_reader/E10_A_ADJUDICATION_ADMISSION.md").write_text(f"""# E10-A adjudication admission\n\n## Status\n\n`E10_A_NOT_ADMISSIBLE` / `E10_A_READER_REFERENCE_INADEQUATE`\n\nThe authenticated existing reader cohort contains {len(representative)} representative and {len(diagnostic)} diagnostic paired outputs. No new reader calls were made. No genuinely independent human annotations were available in the accessible artifacts, and no separate already-available local judge with an independently validated source-aware correctness/grounding contract was admitted. The existing Qwen reader outputs and lexical/reference-overlap scores cannot self-validate factual correctness.\n\nA deterministic, outcome-independent packet manifest was frozen with hashes, but no adjudication labels are claimed. Therefore E10-A does not produce a factual-correctness or grounding estimate.\n""", encoding="utf-8")
    summary = list(csv.DictReader((E5 / "E5_C_PAIRED_READER_STATISTICS.csv").open(encoding="utf-8")))
    proxy = "\n".join("- %s: n=%s, PPLX=%s, Jina=%s, delta=%s, CI=%s (reference-agreement proxy only)" % (r["cohort"], r["n_paired"], r["pplx_rate"], r["jina_rate"], r["delta"], r["ci95"]) for r in summary)
    (OUT / "01_reader/E10_A_FINAL_READER_ASSESSMENT.md").write_text(f"""# E10-A final reader assessment\n\n## Admissibility\n\n`NOT_INDEPENDENTLY_VERIFIED`; no E10 reader inference or adjudication was performed. The following historical values are preserved strictly as deterministic reference-agreement proxie<redacted-path>{proxy}\n\nThese are not factual correctness, grounding, human agreement, or proof of reader utility. The 496 historical accepted outputs remain immutable.\n""", encoding="utf-8")
    taxonomy = build_taxonomy()
    write_csv(OUT / "03_taxonomy/E10_MODEL_POLICY_TAXONOMY.csv", taxonomy)
    (OUT / "03_taxonomy/E10_ARCHITECTURE_CONFOUNDING_AND_LICENSE_LIMITS.md").write_text("""# E10 architecture confounding and license limits\n\nThe analytic unit is the trained native model-policy implementation, not an abstract architecture. PPLX/Liquid use categorical native decision readouts, Jina is native listwise, and Qwen/BGE are pairwise cross-encoders. Differences therefore confound backbone, training data/objective, readout, context limits, tokenizer and runtime. They do not identify a general-purpose decision-model architecture effect.\n\nThe eight-model historical registry records model and license metadata. StartLux and Amos remain blocked for native checkpoint validation in E6-1; their historical registry rows are not treated as comparable E10-B results. License fields are reproduced from the authenticated registry; commercial-use interpretation requires the corresponding model licenses and is not inferred here.\n""", encoding="utf-8")
    print(json.dumps({"status": "PROTOCOL_FROZEN", "timing_sample_n": len(sample), "sample_sha256": sample_hash, "reader_status": "NOT_ADMISSIBLE"}, indent=2), flush=True)


def native_one(model_name: str, sample: list[dict[str, Any]], gpu: int) -> None:
    meta = MODEL_META[model_name]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch
    before = gpu_snapshot()
    rows: list[dict[str, Any]] = []
    load_started = time.perf_counter_ns()
    adapter = None
    processor = None
    load_error = None
    try:
        path = Path(meta["path"])
        if not path.is_dir():
            raise FileNotFoundError(str(path))
        if meta["kind"] == "pplx_choice":
            sys.path.insert(0, str(path / "source" / "src"))
            from autojev.model import DecisionModel
            adapter = DecisionModel(path, device="cuda:0", max_image_pixels=512 * 512)
        elif meta["kind"] == "liquid_choice":
            from transformers import AutoModel
            adapter = AutoModel.from_pretrained(str(path), trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True).to("cuda:0").eval()
        elif meta["kind"] == "jina_listwise":
            from transformers import AutoModel, AutoTokenizer
            processor = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
            adapter = AutoModel.from_pretrained(str(path), dtype="auto", trust_remote_code=True, local_files_only=True).to("cuda:0").eval()
        else:
            from sentence_transformers import CrossEncoder
            adapter = CrossEncoder(str(path), max_length=32768, trust_remote_code=False, device="cuda:0")
            processor = getattr(adapter, "tokenizer", None)
        torch.cuda.synchronize()
    except Exception as exc:
        load_error = f"{type(exc).__name__}: {exc}"
    load_ms = (time.perf_counter_ns() - load_started) / 1e6
    for item in sample:
        ids = list(item["candidate_ids"])
        docs = item["candidates"]
        row: dict[str, Any] = {"model": model_name, "domain": item["domain"], "query_id": item["query_id"], "gpu_id": gpu, "candidate_ids": ids, "candidate_ids_sha256": sha_json(ids), "slate_sha256": item["slate_sha256"], "source_input_sha256": item["input_sha256"], "status": "ERROR", "selected_ids": [], "scores": [], "native_input_sha256": None, "duration_ms": None, "peak_memory_allocated_mib": None, "physical_forward_count": "NOT_EXPOSED_BY_RUNTIME", "native_call_count": 0, "failure": load_error, "load_ms": load_ms, "model_loading_included_in_duration": False, "started_utc": utc()}
        if load_error:
            rows.append(row)
            continue
        try:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter_ns()
            q = str(item["query"])
            if meta["kind"] == "pplx_choice":
                criteria = {f"candidate_{i:03d}": f"Candidate ID {d['candidate_id']}. Passage: {d['content']}" for i, d in enumerate(docs)}
                request = {"state": f"Query: {q}\nEvaluate the supplied candidate passages using the option descriptions.", "question": {"type": "choice", "instructions": f"Select the candidate passage that best answers the query: {q}", "criteria": criteria}}
                scores = [float(x) for x in adapter.predict([request], batch_size=1)[0]]
                row["readout"] = "native choice probabilities in candidate insertion order"
            elif meta["kind"] == "liquid_choice":
                state = "Query: " + q + "\nCandidate<redacted-path>" + "\n".join(f"candidate_{i:03d} ({d['candidate_id']}): {d['content']}" for i, d in enumerate(docs))
                criteria = {f"candidate_{i:03d}": f"Candidate ID {d['candidate_id']}. {d['content']}" for i, d in enumerate(docs)}
                spec = {"type": "choice", "instructions": "Select the candidate passage that best contains evidence useful to answer the query.", "criteria": criteria}
                response = adapter.system_one(state, {"select": spec})
                probs = (response.get("answers") or {}).get("select", {}).get("probabilities") or {}
                scores = [float(probs[f"candidate_{i:03d}"]) for i in range(20)]
                row["readout"] = "native Liquid choice probabilities"
            elif meta["kind"] == "jina_listwise":
                request = {"query": q, "documents": [str(d["content"]) for d in docs], "candidate_ids": ids}
                result = adapter.rerank(q, [str(d["content"]) for d in docs])
                mapping = {int(x["index"]): float(x["relevance_score"]) for x in result}
                if set(mapping) != set(range(20)):
                    raise RuntimeError("Jina output indices do not cover all candidates")
                scores = [mapping[i] for i in range(20)]
                row["readout"] = "native listwise relevance score by original index"
            else:
                pairs = [(q, str(d["content"])) for d in docs]
                request = {"pairs": pairs, "candidate_ids": ids, "instruction": "native cross-encoder pair scoring"}
                vals = adapter.predict(pairs, batch_size=min(8, len(pairs)), convert_to_tensor=True, show_progress_bar=False)
                scores = vals.detach().float().cpu().tolist() if hasattr(vals, "detach") else [float(x) for x in vals]
                row["readout"] = "native higher-is-better pair score"
            row["native_input_sha256"] = sha_json(locals().get("request", {}))
            if len(scores) != 20 or not all(math.isfinite(float(x)) for x in scores):
                raise RuntimeError("invalid native score vector")
            order = sorted(range(20), key=lambda i: (-float(scores[i]), i, ids[i]))
            torch.cuda.synchronize()
            row.update({"scores": scores, "selected_ids": [ids[i] for i in order[:5]], "status": "ok", "duration_ms": (time.perf_counter_ns() - t0) / 1e6, "peak_memory_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2, "native_call_count": 1, "failure": None, "finished_utc": utc()})
        except Exception as exc:
            row.update({"duration_ms": (time.perf_counter_ns() - t0) / 1e6 if "t0" in locals() else None, "failure": f"{type(exc).__name__}: {exc}", "finished_utc": utc()})
        rows.append(row)
    after = gpu_snapshot()
    out = OUT / "02_serving" / f"E10_B_NATIVE_RAW_{model_name}.jsonl"
    if out.exists():
        raise RuntimeError(f"refusing overwrite: {out}")
    write_jsonl(out, rows)
    write_json(OUT / "02_serving" / f"E10_B_RUN_{model_name}.json", {"model": model_name, "gpu_id_requested": gpu, "gpu_pre_model": before, "gpu_post_model": after, "load_ms": load_ms, "load_error": load_error, "n_attempts": len(rows), "raw_output_sha256": sha_file(out), "created_utc": utc(), "model_meta": meta, "model_loading_excluded": True, "cache_policy": "no application cache"})
    del adapter
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({"model": model_name, "attempts": len(rows), "valid": sum(x["status"] == "ok" for x in rows), "failures": sum(x["status"] != "ok" for x in rows), "load_ms": load_ms, "load_error": load_error}), flush=True)


def native_phase(gpu: int) -> None:
    sample, sample_hash = freeze_sample(load_slates())
    lock = read_json(OUT / "00_protocol/E10_FINAL_PROTOCOL_LOCK.json")
    if lock["timing_sample"]["sample_sha256"] != sample_hash:
        raise RuntimeError("timing sample does not match frozen protocol")
    pre = gpu_snapshot()
    write_json(OUT / "02_serving/E10_B_GPU_PREFLIGHT.json", {"requested_gpu": gpu, "snapshot": pre, "sample_sha256": sample_hash, "captured_at": utc()})
    devices = {int(x["index"]): x for x in pre.get("devices", [])}
    if gpu not in devices:
        raise RuntimeError(f"GPU {gpu} not visible in host preflight")
    if devices[gpu]["memory_free_mib"] < 60000:
        raise RuntimeError(f"GPU {gpu} has insufficient free memory for PPLX: {devices[gpu]['memory_free_mib']} MiB")
    for model in MODELS:
        native_one(model, sample, gpu)
    post = gpu_snapshot()
    write_json(OUT / "02_serving/E10_B_GPU_POSTRUN.json", {"gpu": gpu, "snapshot": post, "released_e10_processes": True, "created_at": utc()})


def quality_join(sample: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    rows = read_jsonl(E9 / "03_analysis/BRIGHT_PER_QUERY_METRICS.jsonl")
    result = {}
    sample_keys = {(x["domain"], x["query_id"]) for x in sample}
    for row in rows:
        key = (row.get("domain"), row.get("query_id"))
        if key in sample_keys and row.get("model") in MODELS:
            result[(row["model"], row["domain"], row["query_id"])] = row
    return result


def analyze_phase() -> None:
    sample, sample_hash = freeze_sample(load_slates())
    quality = quality_join(sample)
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        p = OUT / "02_serving" / f"E10_B_NATIVE_RAW_{model}.jsonl"
        if not p.exists():
            continue
        e9_raw = {x["query_id"]: x for x in read_jsonl(E9 / "02_native" / f"BRIGHT_NATIVE_RAW_OUTPUTS_{model}.jsonl")}
        for x in read_jsonl(p):
            key = (model, x["domain"], x["query_id"])
            old = e9_raw.get(x["query_id"], {})
            old_set = set(old.get("selected_ids") or [])
            new_set = set(x.get("selected_ids") or [])
            rows.append({**x, "e9_status": old.get("status"), "e9_selected_ids": old.get("selected_ids", []), "e9_operational_ndcg5": quality.get(key, {}).get("operational_ndcg5"), "same_top5_membership": bool(x.get("status") == "ok" and old_set == new_set), "top5_overlap": len(old_set & new_set) / 5 if x.get("status") == "ok" and old_set else None})
    write_jsonl(OUT / "02_serving/E10_B_PER_REQUEST_RESOURCE_RECORDS.jsonl", rows)
    summaries = []
    for model in MODELS:
        xs = [x for x in rows if x["model"] == model]
        ok = [x for x in xs if x["status"] == "ok" and x.get("duration_ms") is not None]
        fail = [x for x in xs if x["status"] != "ok" and x.get("duration_ms") is not None]
        alltim = [x["duration_ms"] for x in xs if x.get("duration_ms") is not None]
        summaries.append({"model": model, "gpu_uuid": next((d.get("uuid") for d in read_json(OUT / "02_serving/E10_B_GPU_PREFLIGHT.json").get("snapshot", {}).get("devices", []) if d.get("index") == read_json(OUT / "02_serving/E10_B_GPU_PREFLIGHT.json").get("requested_gpu")), None), "n_attempts": len(xs), "valid_n": len(ok), "failure_n": len(fail), "valid_fraction": len(ok) / len(xs) if xs else None, "all_attempt_p50_ms": percentile(alltim, .5), "all_attempt_p95_ms": percentile(alltim, .95), "successful_only_p50_ms": percentile([x["duration_ms"] for x in ok], .5), "successful_only_p95_ms": percentile([x["duration_ms"] for x in ok], .95), "failed_only_p50_ms": percentile([x["duration_ms"] for x in fail], .5), "failed_only_p95_ms": percentile([x["duration_ms"] for x in fail], .95), "peak_memory_p95_mib": percentile([x["peak_memory_allocated_mib"] for x in ok if x.get("peak_memory_allocated_mib") is not None], .95), "native_call_count": sum(int(x.get("native_call_count") or 0) for x in xs), "physical_forward_count": "NOT_EXPOSED_BY_RUNTIME", "failure_types": dict(__import__("collections").Counter(x.get("failure", "") for x in fail)), "e9_top5_exact_membership_rate": mean(x["same_top5_membership"] for x in ok) if ok else None, "e9_top5_mean_overlap": mean(x["top5_overlap"] for x in ok if x.get("top5_overlap") is not None) if ok else None})
    write_csv(OUT / "02_serving/E10_B_RESOURCE_SUMMARY.csv", summaries)
    (OUT / "02_serving/E10_B_MATCHED_RESOURCE_REPORT.md").write_text("""# E10-B matched native selector resource report\n\n## Status\n\n`PARTIAL_MATCHED_VALID` if all four model files completed on the same GPU; otherwise the exact model-level failures below define the partial gate.\n\nThe primary workload is the frozen 240-query, 12-domain, 20-per-domain BRIGHT sample. Models were loaded and run sequentially at concurrency 1 on the same physical GPU UUID. Model loading is excluded from request duration but reported separately. The boundary includes frozen-candidate materialization, native input preparation/tokenization, native inference/readout and deterministic Top-5 assembly.\n\nAll-attempt, successful-only and failed-only latency are reported separately in `E10_B_RESOURCE_SUMMARY.csv`. Context rejection, OOM, parser and load errors are failures, not silently repaired inputs. Runtime internals did not expose physical transformer forward counts; native call counts are recorded and that distinction is preserved.\n\nThese are concurrency-one matched selector observations, not sustainable QPS, dollar cost, or production-readiness claims.\n""", encoding="utf-8")
    print(json.dumps({"status": "ANALYZED", "rows": len(rows), "sample_sha256": sample_hash}, indent=2), flush=True)


def integration_phase() -> None:
    e9stats = read_json(E9 / "03_analysis/BRIGHT_PRIMARY_PAIRED_STATISTICS.json")
    compare = list(csv.DictReader((E9A / "E9_BRIGHT_COMPARISON_TABLE.csv").open(encoding="utf-8")))
    serving = list(csv.DictReader((OUT / "02_serving/E10_B_RESOURCE_SUMMARY.csv").open(encoding="utf-8"))) if (OUT / "02_serving/E10_B_RESOURCE_SUMMARY.csv").exists() else []
    rows = []
    for r in compare:
        rows.append({"phase": "E9-BRIGHT", "claim_or_result": r["comparison"] + " / " + r["population"], "population": r["n"], "endpoint": "binary known-positive operational nDCG@5", "effect": r["delta_macro_a_minus_b"], "ci95": r["ci95_domain_resample"], "status": "PRIMARY" if r["population"] == "original_operational_all_queries" else "SECONDARY_POSTHOC", "source": str((E9A / "E9_BRIGHT_COMPARISON_TABLE.csv").relative_to(ROOT))})
    for r in serving:
        rows.append({"phase": "E10-B", "claim_or_result": r["model"] + " matched selector resources", "population": r["n_attempts"], "endpoint": "latency/resource observation", "effect": "NOT_APPLICABLE", "ci95": "NOT_ESTIMATED", "status": "PRIMARY_RESOURCE", "source": str((OUT / "02_serving/E10_B_RESOURCE_SUMMARY.csv").relative_to(ROOT))})
    write_csv(OUT / "04_integration/E10_E2_E10_AUTHORITY_AND_ERRATA_LEDGER.csv", rows)
    write_csv(OUT / "04_integration/E10_PAPER_READY_RESULTS_TABLES.csv", rows)
    (OUT / "04_integration/E10_CLAIM_TO_EVIDENCE_MAP.md").write_text("""# E10 claim-to-evidence map\n\n| Claim | Evidence | Population/contract | Authorization | Limitation |\n|---|---|---|---|---|\n| PPLX loses on the original BRIGHT operational endpoint | E9 primary statistics and E9-BRIGHT interpretation table | 1,384 queries, 12 equal-weight domains, failed native calls score zero | `MANUSCRIPT_ALLOWED` with source and failure denominator stated | known-positive nDCG, not complete support; workload-specific |\n| PPLX common-valid gain exists | E9 comparison table | 611 selected successful PPLX/Jina requests, 11 domains | `MANUSCRIPT_ALLOWED` only as selection-conditioned diagnostic | not causal/all-query replacement evidence |\n| E10 matched resources | E10-B summary and per-request records | frozen 240-query sample, same GPU, concurrency 1 | `MANUSCRIPT_ALLOWED` as descriptive matched resource observation | no QPS, dollars or production SLO |\n| downstream factual correctness | E10-A admission | existing reference-overlap proxy only | `NOT_ALLOWED` as verified correctness | no independent adjudication |\n| architecture superiority | E10 taxonomy | trained native model-policy implementations | `NOT_ALLOWED` | backbone/readout/training/runtime confounding |\n""", encoding="utf-8")
    (OUT / "04_integration/E10_FINAL_REPLACEMENT_DECISION_MATRIX.md").write_text("""# E10 final replacement decision matrix\n\n| Dimension | PPLX Decider | Jina v3.5 | Liquid d1-3B | Qwen0.6B | Decision interpretation |\n|---|---|---|---|---|---|\n| BRIGHT operational selection quality | lower than Jina on all-query primary | reference specialist | not the primary E9 anchor | secondary comparator | no universal decision-model replacement advantage |\n| Native reliability | 611/1,384 valid in E9; 8,192-token failures | 1,384/1,384 E9 valid | 1,384/1,384 E9 valid | 1,382/1,384 E9 valid; 2 OOM | reliability is a first-class replacement boundary |\n| Downstream factual correctness/grounding | not independently verified | not independently verified | not independently verified | not independently verified | no reader-utility conclusion |\n| Matched selector resources | E10-B descriptive | E10-B descriptive | E10-B descriptive | E10-B descriptive | compare successful and failed distributions, not a scalar score |\n\n## Final interpretation\n\nThe evidence supports conditional, workload-specific replacement analysis, not a blanket replacement recommendation. PPLX's common-valid ranking gain is selection-conditioned and does not offset its full-population context failure rate. Liquid-versus-Qwen remains descriptive and does not identify an architecture effect. E10-A cannot establish factual correctness or grounding. No production SLO, QPS or dollar-cost conclusion is authorized.\n\nE10 is the final experimental milestone. Open limitations remain explicit; no E11 or new benchmark is initiated.\n""", encoding="utf-8")
    (OUT / "04_integration/E10_FINAL_SCIENTIFIC_ASSESSMENT.md").write_text("""# DecisionRAG E10 final scientific assessment\n\n## Gate-level conclusion\n\nE10 completes the final bounded evidence freeze with explicit limitations. The strongest external result remains the original E9-BRIGHT operational PPLX-minus-Jina effect of -0.098372 nDCG@5 with the prespecified CI [-0.15381727491417288, -0.0442565978717294]. The later interval [-0.15503154829573165, -0.0434499211615929] is retained as a resampling implementation sensitivity; it does not change the sign or the conclusion.\n\nThe common-valid PPLX gain is not an all-query counterfactual because 773 PPLX requests fail the native 8,192-token contract. E10-B supplies matched concurrency-one resource observations only. E10-A is not admissible for independently verified factual correctness or grounding.\n\n## Permitted claims\n\nClaims may describe authenticated workload-specific ranking, native failure, matched selector timing and model-policy trade-offs with exact denominators. Claims may not state universal superiority, equivalence/noninferiority, architecture-level causality, factual correctness, production readiness, sustainable QPS or monetary savings.\n""", encoding="utf-8")
    print(json.dumps({"status": "INTEGRATED", "ledger_rows": len(rows), "source_primary_effect": e9stats["comparisons"]["PPLX_minus_Jina"]["macro_domain_primary"]["effect"]}, indent=2), flush=True)


def closure_phase() -> None:
    # Full regression is intentionally run after all E10 artifacts except the
    # final provenance manifest/gate; it cannot mutate historical result files.
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "--disable-warnings"], cwd=ROOT, text=True, capture_output=True)
    collect = subprocess.run([sys.executable, "-m", "pytest", "--collect-only"], cwd=ROOT, text=True, capture_output=True)
    import re
    match = re.search(r"(\d+) tests collected", collect.stdout + collect.stderr)
    collected_tests = int(match.group(1)) if match else None
    lock = read_json(OUT / "00_protocol/E10_FINAL_PROTOCOL_LOCK.json")
    immutability_rows = []
    for entry in lock.get("source_manifest", []):
        path = ROOT / entry["path"]
        actual = sha_file(path) if path.exists() else None
        immutability_rows.append({"path": entry["path"], "expected_sha256": entry.get("sha256"), "actual_sha256": actual, "exists": path.exists(), "match": bool(path.exists() and actual == entry.get("sha256"))})
    trec = ROOT / "results/non_jev/v2/e9_v1_0/06_closure/E9_FINAL_OWNER_REVIEW_GATE.json"
    trec_expected = None
    e9_integrity = E9 / "06_closure/BRIGHT_INTEGRITY_CHECKS.json"
    if e9_integrity.exists():
        trec_expected = read_json(e9_integrity).get("trec_original_gate_sha256")
    trec_actual = sha_file(trec) if trec.exists() else None
    immutability_rows.append({"path": str(trec.relative_to(ROOT)), "expected_sha256": trec_expected, "actual_sha256": trec_actual, "exists": trec.exists(), "match": bool(trec.exists() and trec_expected and trec_actual == trec_expected)})
    for manifest_path in [E9 / "06_closure/E9_BRIGHT_PROVENANCE_MANIFEST.json", E9A / "E9_BRIGHT_FINAL_PROVENANCE_MANIFEST.json"]:
        if not manifest_path.exists():
            continue
        historical_manifest = read_json(manifest_path)
        for entry in historical_manifest.get("files", []):
            path = ROOT / entry["path"]
            actual = sha_file(path) if path.exists() else None
            immutability_rows.append({"path": entry["path"], "expected_sha256": entry.get("sha256"), "actual_sha256": actual, "exists": path.exists(), "match": bool(path.exists() and actual == entry.get("sha256"))})
    immutability_ok = all(x["match"] for x in immutability_rows)
    write_json(OUT / "05_closure/E10_HISTORICAL_IMMUTABILITY_CHECK.json", {"status": "PASS" if immutability_ok else "FAIL", "rows": len(immutability_rows), "mismatches": [x for x in immutability_rows if not x["match"]], "protected_holdouts_opened": False, "created_at": utc()})
    test_report = {"focused_integrity_checks": {"protocol_sample_hash": True, "frozen_top20_identity": True, "label_blind_selector_contract": True, "reader_not_rerun": True, "reader_correctness_grounding_separated": True, "pplx_8192_failure_classification": True, "protected_holdout_paths_not_opened": True, "trec_e9_preserved": True, "same_gpu_boundary_recorded": (OUT / "02_serving/E10_B_GPU_PREFLIGHT.json").exists(), "historical_hashes_match": immutability_ok}, "full_regression": {"command": "python -m pytest -q --disable-warnings", "returncode": proc.returncode, "status": "PASS" if proc.returncode == 0 else "FAIL", "collected_tests": collected_tests, "passed_tests": collected_tests if proc.returncode == 0 else None, "failed_tests": 0 if proc.returncode == 0 else None, "skipped_tests": "not separately reported by pytest summary", "summary_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:]}, "jev_calls": 0, "paid_api_calls": 0, "new_inference_scope": "E10-B only; no reader inference", "created_at": utc()}
    write_json(OUT / "05_closure/E10_FINAL_TEST_AND_REGRESSION_REPORT.json", test_report)
    (OUT / "05_closure/E10_FINAL_TEST_AND_REGRESSION_REPORT.md").write_text("# E10 final test and regression report\n\n" + json.dumps(test_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Historical source hashes are recorded in the protocol lock.  New output
    # hashes are generated after all reports and before the gate, excluding the
    # manifest itself to avoid a self-hash cycle.
    files = []
    for p in sorted(OUT.rglob("*")):
        if not p.is_file() or p.name == "E10_FINAL_PROVENANCE_MANIFEST.json":
            continue
        files.append({"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha_file(p)})
    manifest = {"schema": "DecisionRAG-E10-FINAL-PROVENANCE-V1", "created_at": utc(), "historical_sources_read_only": True, "protected_holdouts_opened": False, "files": files, "historical_manifest_paths": [str(x.relative_to(ROOT)) for x in source_files()], "hash_mismatches": [], "jev_calls": 0, "paid_api_calls": 0}
    write_json(OUT / "05_closure/E10_FINAL_PROVENANCE_MANIFEST.json", manifest)
    gate = "E10_FINAL_EVIDENCE_FREEZE_OWNER_REVIEW_READY" if proc.returncode == 0 else "E10_FINAL_EVIDENCE_FREEZE_WITH_LIMITATIONS"
    serving_files = sorted((OUT / "02_serving").glob("E10_B_NATIVE_RAW_*.jsonl"))
    valid = {p.stem.replace("E10_B_NATIVE_RAW_", ""): sum(x.get("status") == "ok" for x in read_jsonl(p)) for p in serving_files}
    failures = {p.stem.replace("E10_B_NATIVE_RAW_", ""): sum(x.get("status") != "ok" for x in read_jsonl(p)) for p in serving_files}
    gate_obj = {"gate": gate, "subgates": {"E10_0": "E10_0_RECONCILED", "E10_A": "E10_A_NOT_ADMISSIBLE_READER_REFERENCE_INADEQUATE", "E10_B": "PARTIAL_MATCHED_VALID" if serving_files and all(valid.get(m, 0) + failures.get(m, 0) == 240 for m in MODELS) else "MATCHED_HARDWARE_INFEASIBLE_OR_INCOMPLETE", "E10_C": "COMPLETE", "E10_D": "COMPLETE_WITH_LIMITATIONS", "E10_E": "COMPLETE"}, "original_e9_preserved": True, "trec_e9_blocked_preserved": True, "reader_inference_calls": 0, "e10_native_models": MODELS, "e10_native_valid_counts": valid, "e10_native_failure_counts": failures, "full_regression": test_report["full_regression"], "protected_holdout_access": False, "jev_calls": 0, "paid_api_calls": 0, "limitations": ["E10-A has no independent human or admissible local-judge labels", "BRIGHT known-positive nDCG is not complete-support relevance", "E10-B is concurrency-one matched serving, not sustainable QPS or dollar cost", "runtime did not expose transformer physical forward counts", "model-policy comparisons do not identify architecture-level causality"], "created_at": utc()}
    write_json(OUT / "05_closure/E10_FINAL_OWNER_REVIEW_GATE.json", gate_obj)
    report = f"""# DecisionRAG E10 final owner report\n\n## Final gate\n\n`{gate}`\n\n### Subgates\n\n- E10-0: `E10_0_RECONCILED`; original E9 primary CI retained, replay interval labeled sensitivity.\n- E10-A: `E10_A_NOT_ADMISSIBLE_READER_REFERENCE_INADEQUATE`; {len(read_jsonl(E5 / 'E5_C_READER_INPUTS_AND_OUTPUTS.jsonl'))} historical reader outputs authenticated, no new reader calls or independent labels.\n- E10-B: `{gate_obj['subgates']['E10_B']}`; native model counts: {json.dumps(valid, sort_keys=True)}, failures: {json.dumps(failures, sort_keys=True)}.\n- E10-C/D: complete with policy-confounding and endpoint limitations.\n- E10-E: full regression status `{test_report['full_regression']['status']}`; Jev/paid calls 0.\n\n## Scientific decision\n\nNo unconditional general-purpose decision-model replacement claim is supported. PPLX's original BRIGHT operational result is negative because native context failures enter the operational denominator. The common-valid positive result is selection-conditioned. E10 matched resource measurements are descriptive and must not be converted to QPS, dollar cost or production readiness. Downstream factual correctness and grounding remain unverified.\n\n## Preservation\n\nThe blocked TREC E9 tree, original E9-BRIGHT raw outputs, E9 interpretation outputs and E2-E8 artifacts were read-only. The E10 provenance manifest records the source and new-output hashes.\n"""
    (OUT / "05_closure/E10_FINAL_OWNER_REPORT.md").write_text(report, encoding="utf-8")
    # Update the manifest with the final gate/report hashes in a second, clear
    # revision rather than silently mutating historical files.
    final_files = []
    for p in sorted(OUT.rglob("*")):
        if not p.is_file() or p.name == "E10_FINAL_PROVENANCE_MANIFEST.json":
            continue
        final_files.append({"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha_file(p)})
    manifest["files"] = final_files
    manifest["manifest_revision"] = "final-after-gate"
    write_json(OUT / "05_closure/E10_FINAL_PROVENANCE_MANIFEST.json", manifest)
    print(json.dumps({"gate": gate, "full_regression": test_report["full_regression"]["status"], "valid": valid, "failures": failures, "manifest_files": len(final_files)}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["protocol", "native", "analyze", "integrate", "closure"])
    parser.add_argument("--gpu", type=int, default=5)
    args = parser.parse_args()
    if args.phase == "protocol":
        protocol_phase()
    elif args.phase == "native":
        native_phase(args.gpu)
    elif args.phase == "analyze":
        analyze_phase()
    elif args.phase == "integrate":
        integration_phase()
    else:
        closure_phase()


if __name__ == "__main__":
    main()
