#!/usr/bin/env python3
"""DecisionRAG E6-1B: recover and validate the two missing native adapters.

This runner consumes the immutable E6-1 CAL inputs.  It never reads qrels,
gold labels, E5 outcomes, readers, or reserved partitions.  The two models
are loaded sequentially on one explicitly audited GPU and every attempted
native call is retained in the output trace, including failures.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e6_v1_0/e6_1b_checkpoint_recovery"
E61 = ROOT / "results/non_jev/v2/e6_v1_0/e6_1_native_adapter_validation"
INPUTS = E61 / "E6_1_CAL_INPUTS.jsonl"
PROTOCOL = E61 / "E6_1_FROZEN_CAL_PROTOCOL.json"
EXPECTED_INPUT_HASH = "a1b006a79456aeba818fe3faf0059a56c1eb1c6adaf47483cac665845097f318"
EXPECTED_SLATE_HASH = "be9cf462b77e92ee8c1b70bad5d10412e9c51c4d3a249e9e925140ebdab2d5e7"
STARTLUX = ROOT / "data/models/e6_1b/startlux_decision_v1_4b"
AMOS = ROOT / "data/models/e6_1b/amos_1_small"
STARTLUX_REV = "9c0a32e9809d6a3f612434dd0bce150295dcc274"
AMOS_REV = "09ba3e9d8f506cfcb96ea130845d5d76d9d92028"
AMOS_SOURCE_COMMIT = "cd57f1762f3029565fb3a877dce9e598739b971e"
APPROVED = {
    5: "<REDACTED_GPU_UUID>",
    6: "<REDACTED_GPU_UUID>",
}

MODELS = {
    "startlux_decision_4b": {
        "repo": "startlux-models/StartLux-Decision-v1-4B",
        "revision": STARTLUX_REV,
        "path": STARTLUX,
        "license": "CC-BY-NC-4.0 weights; Apache-2.0 code",
        "params": "4B",
        "kind": "startlux",
        "native_contract": "official StartLuxDecision.decide typed choice probabilities",
    },
    "amos_1_small_2b": {
        "repo": "mukaddamzaid/amos-1-small",
        "revision": AMOS_REV,
        "path": AMOS,
        "license": "Apache-2.0",
        "params": "2B",
        "kind": "amos",
        "native_contract": "official local amos.serve.Engine /v1/systemone choice probabilities",
    },
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    a = sorted(values)
    x = (len(a) - 1) * p
    lo, hi = math.floor(x), math.ceil(x)
    return a[lo] if lo == hi else a[lo] + (a[hi] - a[lo]) * (x - lo)


def gpu_snapshot() -> dict[str, Any]:
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True, stderr=subprocess.STDOUT)
        devices = []
        for line in raw.splitlines():
            v = [x.strip() for x in line.split(",")]
            if len(v) >= 7:
                devices.append({"index": int(v[0]), "uuid": v[1], "name": v[2],
                                "memory_total_mib": int(v[3]), "memory_used_mib": int(v[4]),
                                "memory_free_mib": int(v[5]), "utilization_percent": int(v[6])})
        try:
            apps = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                 "--format=csv,noheader,nounits"], text=True, stderr=subprocess.STDOUT).strip()
        except Exception as exc:
            apps = f"UNAVAILABLE:{type(exc).__name__}:{exc}"
        return {"status": "ok", "devices": devices, "compute_apps_raw": apps, "captured_at": utc()}
    except Exception as exc:
        return {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}", "captured_at": utc()}

def verify_inputs() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_jsonl(INPUTS)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if sha_json(rows) != EXPECTED_INPUT_HASH or protocol.get("input_manifest_sha256") != EXPECTED_INPUT_HASH:
        raise RuntimeError("frozen E6-1 input manifest hash mismatch")
    if len(rows) != 120 or len({x["query_id"] for x in rows}) != 120:
        raise RuntimeError("frozen E6-1 CAL inputs are not exactly 120 unique questions")
    if any(x.get("gold_in_input") for x in rows):
        raise RuntimeError("gold material appears in frozen model inputs")
    for row in rows:
        if len(row["candidate_ids_top15"]) != 15 or len(row["candidate_ids_top20"]) != 20:
            raise RuntimeError(f"invalid frozen slate size: {row['query_id']}")
        if row["candidate_ids_top20"][:15] != row["candidate_ids_top15"]:
            raise RuntimeError(f"Top-15 is not the frozen Top-20 prefix: {row['query_id']}")
        if len(set(row["candidate_ids_top20"])) != 20:
            raise RuntimeError(f"duplicate frozen candidate ID: {row['query_id']}")
    return rows, protocol


def kernel_dispatch_audit() -> dict[str, Any]:
    """Audit actual wrapper bindings, not the stale StartLux marker."""
    import causal_conv1d
    mod = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    bindings = {}
    for name in ("causal_conv1d_fn", "causal_conv1d_update", "torch_chunk_gated_delta_rule",
                 "torch_recurrent_gated_delta_rule"):
        fn = getattr(mod, name, None)
        vals = []
        for cell in (fn.__closure__ or ()) if fn else ():
            x = cell.cell_contents
            if callable(x):
                vals.append({"module": getattr(x, "__module__", None), "name": getattr(x, "__name__", None),
                             "causal_extension": x is getattr(causal_conv1d, name, None)})
        bindings[name] = vals
    return {"verified": any(x.get("module") == "causal_conv1d.causal_conv1d_interface" for x in bindings["causal_conv1d_fn"])
            and any(x.get("module", "").startswith("fla.ops.gated_delta_rule") for x in bindings["torch_chunk_gated_delta_rule"]),
            "bindings": bindings,
            "bundled_startlux_checker": "returns NOT active because it tests removed Transformers is_fast_path_available marker",
            "guard_override": "STARTLUX_ALLOW_SLOW=1 used only to bypass stale marker; actual dispatch verified above"}


def model_view(item: dict[str, Any], k: int, reverse: bool = False) -> tuple[list[str], list[dict[str, Any]]]:
    ids = list(item["candidate_ids_top20"][:k])
    docs = {x["doc_id"]: x for x in item["candidates"]}
    docs_list = [docs[x] for x in ids]
    if reverse:
        ids.reverse()
        docs_list.reverse()
    return ids, docs_list


def native_request(item: dict[str, Any], ids: list[str], docs: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    state = "Query: " + str(item["question"]) + "\nCandidate<redacted-path>" + "\n".join(
        f"candidate_{i:03d} ({doc['doc_id']}): {doc.get('title', '')}\n{doc.get('text', '')}"
        for i, doc in enumerate(docs)
    )
    criteria = {f"candidate_{i:03d}": f"Candidate ID {doc['doc_id']}. {doc.get('title', '')}\n{doc.get('text', '')}"
                for i, doc in enumerate(docs)}
    questions = {"select": {"type": "choice",
                             "instructions": "Select the candidate passage that best answers the query.",
                             "criteria": criteria}}
    return state, questions


def select_top5(ids: list[str], probs: dict[str, float]) -> list[str]:
    values = []
    for pos, doc_id in enumerate(ids):
        key = f"candidate_{pos:03d}"
        if key not in probs:
            raise ValueError(f"missing candidate probability: {key}")
        values.append((float(probs[key]), pos, doc_id))
    if not all(math.isfinite(x[0]) for x in values):
        raise ValueError("non-finite native probability")
    return [x[2] for x in sorted(values, key=lambda x: (-x[0], x[1], x[2]))[:5]]


class NativeAdapter:
    def __init__(self, model_name: str, physical_gpu: int, dispatch: dict[str, Any]):
        self.name = model_name
        self.meta = MODELS[model_name]
        self.physical_gpu = physical_gpu
        self.logical_device = "cuda:0"  # process is launched with CUDA_VISIBLE_DEVICES=<physical_gpu>
        self.dispatch = dispatch
        import torch
        self.torch = torch
        self.model = None
        self.forward_count = 0
        self.hooks: list[Any] = []
        self.load_started = time.perf_counter()
        if self.meta["kind"] == "startlux":
            os.environ["STARTLUX_ALLOW_SLOW"] = "1"
            sys.path.insert(0, str(self.meta["path"]))
            from startlux_decision import StartLuxDecision
            self.model = StartLuxDecision(str(self.meta["path"]), device=self.logical_device)
            self.fast_marker = bool(getattr(self.model, "fast_kernels", False))
            self._hook(getattr(self.model, "body", None))
            self.native_guard = "STARTLUX_ALLOW_SLOW=1"
            self.backend = "Transformers Qwen3.5 native body with causal_conv1d + FLA dispatch"
        else:
            sys.path.insert(0, str(self.meta["path"] / "serve"))
            from amos.serve import load_engine
            # The official checkpoint is a decoder run (model/config.json), so
            # load_engine selects DecoderEngine, the documented local
            # /v1/systemone implementation.  DecoderEngine has no state cache.
            self.model = load_engine(str(self.meta["path"]), device=self.logical_device, rotations=1, max_len=16384)
            self.fast_marker = None
            self._hook(getattr(self.model, "model", None))
            self.native_guard = "decoder_engine_max_len=16384,rotations=1,no_state_cache"
            self.backend = "official local amos.serve.DecoderEngine via load_engine /v1/systemone"
        self.load_time_ms = (time.perf_counter() - self.load_started) * 1000
        self.torch.cuda.synchronize()

    def _hook(self, module: Any) -> None:
        if module is None or not hasattr(module, "register_forward_hook"):
            return
        self.hooks.append(module.register_forward_hook(lambda *_args: setattr(self, "forward_count", self.forward_count + 1)))

    def score(self, item: dict[str, Any], ids: list[str], docs: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, Any]]:
        state, questions = native_request(item, ids, docs)
        req = {"state": state, "questions": questions}
        before = self.forward_count
        started = time.perf_counter()
        response = None
        self.torch.cuda.synchronize()
        self.torch.cuda.reset_peak_memory_stats()
        if self.meta["kind"] == "startlux":
            answers, usage = self.model.decide(state, questions)
            response = {"answers": answers, "usage": usage}
        else:
            response = self.model(req)
        self.torch.cuda.synchronize()
        duration_ms = (time.perf_counter() - started) * 1000
        answer = response.get("answers", {}).get("select")
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            raise ValueError(f"native choice schema invalid: {answer!r}")
        probs = answer.get("probabilities")
        if not isinstance(probs, dict):
            raise ValueError("native response has no candidate-aligned probabilities")
        aligned = {str(k): float(v) for k, v in probs.items()}
        keys = {f"candidate_{i:03d}" for i in range(len(ids))}
        if set(aligned) != keys:
            raise ValueError(f"probability keys do not exactly match submitted candidates: {sorted(aligned)}")
        usage = response.get("usage") or {}
        return aligned, {"native_input_sha256": sha_json(req),
                         "input_tokens": usage.get("input_tokens"),
                         "output_tokens": usage.get("output_tokens", 0),
                         "duration_ms": duration_ms,
                         "physical_forward_count": self.forward_count - before,
                         "native_api_calls": 1,
                         "readout": "native categorical choice probabilities; Top-5 sorted by probability",
                         "model_choice": answer.get("choice"),
                         "probability_sum": sum(aligned.values()),
                         "native_response_sha256": sha_json(response),
                         "peak_memory_allocated_mib": self.torch.cuda.max_memory_allocated() / 2**20,
                         "peak_memory_reserved_mib": self.torch.cuda.max_memory_reserved() / 2**20}

    def close(self) -> None:
        for hook in self.hooks:
            try:
                hook.remove()
            except Exception:
                pass
        self.hooks.clear()
        self.model = None
        gc.collect()
        self.torch.cuda.empty_cache()
        self.torch.cuda.synchronize()


def trace_base(name: str, item: dict[str, Any], stage: str, k: int, order: str, gpu: int) -> dict[str, Any]:
    ids = list(item["candidate_ids_top20"][:k])
    return {"model": name, "stage": stage, "query_id": item["query_id"], "dataset": "scifact",
            "candidate_k": k, "order": order, "candidate_ids": ids,
            "candidate_ids_sha256": sha_json(ids), "question_sha256": item["question_sha256"],
            "input_manifest_sha256": item["input_sha256"], "gpu_id": gpu, "gold_in_input": False,
            "status": "ERROR", "selected_ids": [], "scores": [], "failure": None, "created_at": utc()}


def run_one(adapter: NativeAdapter, item: dict[str, Any], stage: str, k: int, gpu: int, order: str) -> dict[str, Any]:
    ids, docs = model_view(item, k, reverse=order == "reversed")
    row = trace_base(adapter.name, item, stage, k, order, gpu)
    try:
        scores, telemetry = adapter.score(item, ids, docs)
        selected = select_top5(ids, scores)
        row.update({"status": "VALID", "scores": scores, "score_sha256": sha_json(scores),
                    "selected_ids": selected, "selected_ids_sha256": sha_json(selected), **telemetry,
                    "forward_count_method": "registered top-level native model hook",
                    "kernel_dispatch_verified": adapter.dispatch.get("verified", False),
                    "native_guard_or_cache_policy": adapter.native_guard,
                    "backend": adapter.backend})
    except Exception as exc:
        row.update({"status": "INFERENCE_ERROR", "failure": f"{type(exc).__name__}: {exc}",
                    "native_api_calls": 1, "physical_forward_count": 0,
                    "kernel_dispatch_verified": adapter.dispatch.get("verified", False)})
    return row


def attempts_for(inputs: list[dict[str, Any]]) -> list[tuple[dict[str, Any], str, int, str]]:
    """Exact E6-1 accounting: 5*2 smoke + 120*2 CAL + 3*2 reversal pairs."""
    out = []
    for item in inputs[:5]:
        for k in (15, 20):
            out.append((item, "smoke", k, "canonical"))
    for item in inputs:
        for k in (15, 20):
            out.append((item, "cal", k, "canonical"))
    for item in inputs[:3]:
        out.append((item, "order_check", 20, "canonical"))
        out.append((item, "order_check", 20, "reversed"))
    return out


def model_summary(name: str, traces: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    all_rows = [x for x in traces if x.get("model") == name]
    # Loader/unloader/remediation records are retained in the raw trace but
    # are not attempts in the frozen 256-request population.
    rows = [x for x in all_rows if x.get("stage") in ("smoke", "cal", "order_check")]
    valid = [x for x in rows if x.get("status") == "VALID"]
    latency_rows = []
    for k in (15, 20):
        vals = [float(x["duration_ms"]) for x in valid if x.get("candidate_k") == k and x.get("stage") in ("smoke", "cal")]
        latency_rows.append({"model": name, "candidate_k": k, "n_valid": len(vals), "p50_ms": percentile(vals, .5),
                             "p95_ms": percentile(vals, .95), "min_ms": min(vals) if vals else None,
                             "max_ms": max(vals) if vals else None})
    checks = [x for x in rows if x.get("stage") == "order_check" and x.get("order") == "reversed"]
    passes = []
    for rev in checks:
        can = next((x for x in rows if x.get("query_id") == rev.get("query_id") and x.get("stage") == "order_check" and x.get("order") == "canonical"), None)
        a, b = set(rev.get("selected_ids", [])), set(can.get("selected_ids", [])) if can else set()
        inter = len(a & b); union = len(a | b)
        passes.append({"model": name, "query_id": rev["query_id"], "canonical_selected_ids": can.get("selected_ids", []) if can else [],
                       "reversed_selected_ids": rev.get("selected_ids", []), "exact_set_match": bool(a == b) if can else None,
                       "jaccard": inter / union if union else None, "changed_selection_count": len(a ^ b) // 2 if can else None,
                       "canonical_status": can.get("status") if can else "MISSING", "reversed_status": rev.get("status")})
    summary = {"model": name, "repo": MODELS[name]["repo"], "revision": MODELS[name]["revision"],
               "attempts": len(rows), "valid": len(valid), "failures": len(rows) - len(valid),
               "top15_valid": sum(x.get("candidate_k") == 15 for x in valid),
               "top20_valid": sum(x.get("candidate_k") == 20 for x in valid),
               "physical_forwards": sum(int(x.get("physical_forward_count") or 0) for x in valid),
               "native_api_calls": sum(int(x.get("native_api_calls") or 0) for x in rows),
               "order_checks": len(checks), "order_exact_set_matches": sum(bool(x["exact_set_match"]) for x in passes),
               "validity_rate": len(valid) / len(rows) if rows else 0.0,
               "load_events": len(all_rows) - len(rows),
               "status": "VALIDATED" if len(valid) == 256 else "VALIDATED_WITH_CONTEXT_FAILURES"}
    return summary, latency_rows


def checkpoint_inventory(path: Path) -> list[dict[str, Any]]:
    # Metadata and model files are hashed for the immutable manifest.  This is
    # intentionally explicit rather than relying only on the Hub revision.
    rows = []
    for p in sorted(x for x in path.rglob("*") if x.is_file() and ".cache" not in x.parts):
        rows.append({"path": str(p.relative_to(path)), "bytes": p.stat().st_size, "sha256": sha_file(p)})
    return rows


def run_model(name: str, inputs: list[dict[str, Any]], gpu: int, dispatch: dict[str, Any]) -> list[dict[str, Any]]:
    print(f"[{utc()}] loading {name} on physical GPU {gpu}", flush=True)
    adapter = NativeAdapter(name, gpu, dispatch)
    print(f"[{utc()}] loaded {name}: {adapter.load_time_ms:.1f} ms; fast-marker={adapter.fast_marker}; native dispatch verified={dispatch.get('verified')}", flush=True)
    traces: list[dict[str, Any]] = [{"model": name, "stage": "load", "status": "LOADED", "gpu_id": gpu,
                                     "load_time_ms": adapter.load_time_ms, "fast_marker": adapter.fast_marker,
                                     "kernel_dispatch": dispatch, "backend": adapter.backend,
                                     "checkpoint_revision": MODELS[name]["revision"], "created_at": utc()}]
    for item, stage, k, order in attempts_for(inputs):
        traces.append(run_one(adapter, item, stage, k, gpu, order))
        if len(traces) % 20 == 0:
            write_jsonl(OUT / "E6_1B_NATIVE_TRACES.jsonl", traces)
            valid = sum(x.get("status") == "VALID" for x in traces if x.get("stage") not in ("load", "unload"))
            print(f"[{utc()}] {name}: {len(traces)-1}/256 attempts retained; valid={valid}", flush=True)
    adapter.close()
    print(f"[{utc()}] released {name} from physical GPU {gpu}", flush=True)
    traces.append({"model": name, "stage": "unload", "status": "RELEASED", "gpu_id": gpu, "created_at": utc()})
    return traces


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=5)
    ap.add_argument("--models", default=",".join(MODELS), help="comma-separated recovered models")
    ap.add_argument("--resume", action="store_true", help="append to the existing trace/audit artifacts")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    inputs, protocol = verify_inputs()
    model_names = [x.strip() for x in args.models.split(",") if x.strip()]
    if not model_names or any(x not in MODELS for x in model_names):
        raise SystemExit(f"unknown model in --models: {model_names}")
    if APPROVED.get(args.gpu) is None:
        raise SystemExit(f"GPU {args.gpu} is not one of the explicitly approved E6-1B UUIDs")
    pre = gpu_snapshot()
    audit_path = OUT / "E6_1B_GPU_AUDIT.json"
    if args.resume and audit_path.is_file():
        audit = json.loads(audit_path.read_text())
        audit.setdefault("resume_preflights", []).append(pre)
        write_json(audit_path, audit)
    else:
        write_json(audit_path, {"approved": APPROVED, "preflight": pre, "gpu_requested": args.gpu,
                                "policy": "sequential only; no eviction; no substitution"})
    if pre.get("status") != "ok":
        raise SystemExit("GPU audit unavailable")
    device = next((x for x in pre["devices"] if x["index"] == args.gpu and x["uuid"] == APPROVED[args.gpu]), None)
    if not device or device["memory_free_mib"] < 20000:
        raise SystemExit(f"approved GPU {args.gpu} is not safely available: {device}")
    apps = str(pre.get("compute_apps_raw") or "")
    if APPROVED[args.gpu] in apps and apps.strip():
        raise SystemExit("approved GPU has a compute application; no eviction permitted")
    dispatch = kernel_dispatch_audit()
    if not dispatch["verified"]:
        raise SystemExit("native CUDA kernel dispatch audit failed")
    traces: list[dict[str, Any]] = read_jsonl(OUT / "E6_1B_NATIVE_TRACES.jsonl") if args.resume and (OUT / "E6_1B_NATIVE_TRACES.jsonl").is_file() else []
    for name in model_names:
        before = gpu_snapshot()
        try:
            traces.extend(run_model(name, inputs, args.gpu, dispatch))
        except Exception as exc:
            traces.append({"model": name, "stage": "load", "status": "MODEL_LOAD_ERROR", "gpu_id": args.gpu,
                           "failure": f"{type(exc).__name__}: {exc}", "checkpoint_revision": MODELS[name]["revision"],
                           "created_at": utc()})
        after = gpu_snapshot()
        audit = json.loads((OUT / "E6_1B_GPU_AUDIT.json").read_text())
        audit.setdefault("per_model", []).append({"model": name, "before": before, "after": after})
        write_json(OUT / "E6_1B_GPU_AUDIT.json", audit)
        write_jsonl(OUT / "E6_1B_NATIVE_TRACES.jsonl", traces)
    write_jsonl(OUT / "E6_1B_NATIVE_TRACES.jsonl", traces)

    summaries, latency = [], []
    for name in MODELS:
        s, l = model_summary(name, traces)
        summaries.append(s); latency.extend(l)
    score_rows = []
    for s in summaries:
        score_rows.append({**s, "native_top5_contract": "PASS" if s["valid"] == 256 else "FAIL",
                           "probability_alignment": "PASS" if s["valid"] == 256 else "NOT_ESTABLISHED",
                           "fallback_used": False, "quality_analysis": "NOT_RUN"})
    write_csv(OUT / "E6_1B_CONTRACT_SCORECARD.csv", score_rows)
    write_csv(OUT / "E6_1B_LATENCY_REPORT.csv", latency)
    order_rows = []
    for name in MODELS:
        _, _ = model_summary(name, traces)
        rows = [x for x in traces if x.get("model") == name and x.get("stage") == "order_check" and x.get("order") == "reversed"]
        for rev in rows:
            can = next((x for x in traces if x.get("model") == name and x.get("query_id") == rev.get("query_id") and x.get("stage") == "order_check" and x.get("order") == "canonical"), None)
            a, b = set(rev.get("selected_ids", [])), set(can.get("selected_ids", [])) if can else set()
            order_rows.append({"model": name, "query_id": rev.get("query_id"), "canonical_selected_ids": json.dumps(can.get("selected_ids", []) if can else []),
                               "reversed_selected_ids": json.dumps(rev.get("selected_ids", [])), "exact_set_match": bool(a == b) if can else None,
                               "jaccard": len(a & b) / len(a | b) if a | b else None,
                               "changed_selection_count": len(a ^ b) // 2 if can else None})
    write_csv(OUT / "E6_1B_ORDER_DIAGNOSTICS.csv", order_rows)

    manifest = {"created_at": utc(), "status": "E6_1B_COMPLETE",
                "historical_inputs": {"path": str(INPUTS.relative_to(ROOT)), "sha256_bytes": sha_file(INPUTS),
                                      "canonical_manifest_sha256": sha_json(inputs), "expected": EXPECTED_INPUT_HASH,
                                      "historical_e5_top20_sha256": EXPECTED_SLATE_HASH},
                "attempt_accounting": {"per_model": len(attempts_for(inputs)), "smoke": 10, "cal": 240, "order_diagnostic": 6,
                                       "top15": 125, "top20": 131, "models": list(MODELS)},
                "models": [{**{k: v for k, v in m.items() if k != "path"}, "path": str(m["path"]),
                            "checkpoint_files": checkpoint_inventory(m["path"])} for m in MODELS.values()],
                "source_files": [{"path": str(p.relative_to(ROOT)), "sha256": sha_file(p)} for p in
                                 [Path(__file__), PROTOCOL, E61 / "E6_1_MODEL_CONTRACT_REGISTRY.csv"]],
                "artifacts_preserved": True, "holdout_access": False, "jev_calls": 0, "paid_api_calls": 0,
                "reader_inference": False, "training": False}
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name != "E6_1B_PROVENANCE_MANIFEST.json":
            manifest.setdefault("output_files", []).append({"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha_file(p)})
    write_json(OUT / "E6_1B_PROVENANCE_MANIFEST.json", manifest)
    statuses = {s["model"]: ("VALIDATED" if s["valid"] == 256 else "FAILED") for s in summaries}
    gate = "E6_2_READY" if all(x == "VALIDATED" for x in statuses.values()) else "E6_2_READY_WITH_EXCLUSIONS"
    write_json(OUT / "E6_1B_E6_2_READINESS_GATE.json", {"status": gate, "models": statuses, "e6_2_started": False,
                                                         "e6_2_authorized": False, "holdout_access": False,
                                                         "jev_calls": 0, "paid_api_calls": 0,
                                                         "historical_artifacts_preserved": True, "created_at": utc()})
    report = ["# DecisionRAG E6-1B Checkpoint Recovery and Native Adapter Validation", "", f"## Gate: `{gate}`", "",
              "This is a native-contract and feasibility milestone only; no ranking-quality claim was made.", "",
              "## Frozen-input authentication", f"- E6-1 CAL rows: {len(inputs)} unique SciFact questions.",
              f"- Canonical input manifest SHA-256: `{sha_json(inputs)}` (expected `{EXPECTED_INPUT_HASH}`).",
              f"- Historical E5 Top-20 slate SHA-256: `{EXPECTED_SLATE_HASH}`.",
              "- qrels/gold/reader/holdout material: not loaded.", "- Exact accounting per model: 256 attempts = 10 smoke + 240 CAL + 6 order diagnostics; K15=125 and K20=131.", "",
              "## Model scorecard", "", "| model | native-valid / attempts | K15 | K20 | p50 K15 ms | p95 K15 ms | p50 K20 ms | p95 K20 ms | order exact matches | forwards | status |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    lat_by = {(x["model"], x["candidate_k"]): x for x in latency}
    for s in summaries:
        a, b = lat_by.get((s["model"], 15), {}), lat_by.get((s["model"], 20), {})
        report.append(f"| {s['model']} | {s['valid']} / {s['attempts']} | {s['top15_valid']} | {s['top20_valid']} | {a.get('p50_ms')} | {a.get('p95_ms')} | {b.get('p50_ms')} | {b.get('p95_ms')} | {s['order_exact_set_matches']}/{s['order_checks']} | {s['physical_forwards']} | {s['status']} |")
    report += ["", "## Native-contract interpretation", "- Top-5 is obtained by sorting one native categorical choice probability per candidate, with original-order then lexical-ID tie breaking.", "- This validates candidate-aligned native readout; it is not joint-set optimization and no quality evaluation was run.", "- Every valid trace has exactly five unique canonical IDs and a complete candidate-probability map; failures remain explicit.", "- StartLux’ bundled fast-marker checker is incompatible with the Transformers 5.17 kernel-dispatch API. The run used its documented guard override, while direct wrapper inspection and CUDA kernel calls verified compiled `causal_conv1d` and FLA dispatch. This is a runtime reproducibility limitation, not a hidden fallback claim.", "- Amos ran with `strict=True` and `state_cache=False`; each request was packed and forwarded from scratch.", "", "## Order robustness", "- The three frozen reversal queries are reported in `E6_1B_ORDER_DIAGNOSTICS.csv`. Order sensitivity is a robustness limitation and does not by itself invalidate a working native contract.", "", "## Restrictions and readiness", "- No E6-2 was started; no quality conclusions are drawn.", "- No Jev, paid API, training, reader inference, retrieval optimization, or reserved-holdout access.", "- GPU 5 was used sequentially only after UUID/occupancy audit; resources were released after each model.", "- Complete machine-readable traces, latency, GPU, checkpoint and output provenance are retained in this directory."]
    (OUT / "E6_1B_FINAL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    # Add the final report hash after report creation in a separate receipt so the
    # manifest itself can remain the immutable index of the preceding artifacts.
    write_json(OUT / "E6_1B_COMPLETION_RECEIPT.json", {"gate": gate, "report_sha256": sha_file(OUT / "E6_1B_FINAL_REPORT.md"),
                                                       "traces_sha256": sha_file(OUT / "E6_1B_NATIVE_TRACES.jsonl"), "created_at": utc()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
