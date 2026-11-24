#!/usr/bin/env python3
"""One label-blind PPLX native feasibility request for v1.3 preflight."""
from __future__ import annotations
import gc, hashlib, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E11 = ROOT / "results/non_jev/v2/e11_v1_0"
OUT = ROOT / "results/v1_3/preflight/native_smoke"
PPLX_REV = "14c9d90137962d300f3de95633f7b6535e80c5ce"
PPLX_PATH = Path(f"<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--perplexity-ai--pplx-decider-v1.1-27b/snapshots/{PPLX_REV}")

def sha_obj(x):
    return hashlib.sha256(json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def query_gpu(index):
    try:
        raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.free,memory.used", "--format=csv,noheader,nounits"], text=True)
        rows = []
        for line in raw.splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) == 4 and int(p[0]) == index:
                rows.append({"index": int(p[0]), "uuid": p[1], "memory_free_mib": int(p[2]), "memory_used_mib": int(p[3])})
        return rows[0] if rows else {"error": "GPU_NOT_FOUND"}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}:{exc}"}

def main():
    gpu = int(os.environ.get("V13_GPU", "0"))
    before = query_gpu(gpu)
    OUT.mkdir(parents=True, exist_ok=True)
    row = json.loads(next(x for x in (E11 / "01_budget_policy/E11_A_FROZEN_INPUT_AND_TOKEN_MANIFEST.jsonl").open(encoding="utf-8") if x.strip()))
    record = {"stage": "preflight_native_smoke", "policy": "pplx_budget", "gpu": gpu, "gpu_before": before, "model_revision": PPLX_REV, "source_row": {k: row[k] for k in ("domain", "query_id", "slate_sha256", "pplx_request_sha256", "budgeted_candidate_ids")}, "labels_joined": False}
    if before.get("memory_free_mib", 0) < 60000:
        record.update({"status": "BLOCKED_GPU_MEMORY_GATE", "error": "GPU free memory below 60000 MiB; no model load attempted"})
        (OUT / "SMOKE_RESULT_ATTEMPT_02.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2)); return
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        sys.path.insert(0, str(PPLX_PATH / "source" / "src"))
        import torch
        from autojev.model import DecisionModel, decision_messages
        adapter = DecisionModel(PPLX_PATH, device="cuda:0", max_image_pixels=512 * 512)
        codes = [f"candidate_{i:03d}" for i in range(20)]
        criteria = {codes[i]: f"Candidate ID {d['candidate_id']}. Passage: {d['content']}" for i, d in enumerate(row["budgeted_candidates"])}
        request = {"state": f"Query: {row['query']}\nEvaluate the supplied candidate passages using the option descriptions.", "question": {"type": "choice", "instructions": f"Select the candidate passage that best answers the query: {row['query']}", "criteria": criteria}}
        t = time.perf_counter()
        scores = [float(x) for x in adapter.predict([request], batch_size=1)[0]]
        torch.cuda.synchronize()
        order = sorted(range(20), key=lambda i: (-scores[i], i))
        record.update({"status": "PASS", "elapsed_ms": (time.perf_counter() - t) * 1000, "scores_sha256": sha_obj(scores), "selected_ids": [row["budgeted_candidate_ids"][i] for i in order[:5]], "native_request_sha256": sha_obj(request), "peak_memory_allocated_mib": torch.cuda.max_memory_allocated() / 2**20, "score_count": len(scores), "valid_top5": len(set(row["budgeted_candidate_ids"][i] for i in order[:5])) == 5})
        del adapter
        gc.collect(); torch.cuda.empty_cache()
    except Exception as exc:
        record.update({"status": "FAIL", "error": f"{type(exc).__name__}:{exc}"})
    record["gpu_after"] = query_gpu(gpu)
    (OUT / "SMOKE_RESULT_ATTEMPT_02.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))

if __name__ == "__main__":
    main()
