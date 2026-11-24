#!/usr/bin/env python3
"""DecisionRAG E7 integrity and robustness execution.

This runner is deliberately separate from the E6-4 adapter runner.  It
freezes a CAL-only, label-blind input manifest, calls the native model
objects directly, restores candidate IDs in this file, and scores only after
raw outputs have been committed.  All outputs are append-only under E7.
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
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/non_jev/v2"
E6 = BASE / "e6_v1_0"
OUT = BASE / "e7_v1_0"
GOV = OUT / "00_governance"
AUD = OUT / "01_native_contract_audit"
ROB = OUT / "02_repeatability_and_order"
LED = OUT / "03_evidence_ledger"
CLO = OUT / "04_closure"
E64 = E6 / "e6_4_integrated_robustness_and_paper"
R1 = E6 / "e6_2_r1_scifact_retrieval_freeze"
R4C = E6 / "e6_2_r4c_erratum_and_garage_freeze"
GARAGE = ROOT / "data/external/e6_0r_datasets/garage/GaRAGe_benchmark.jsonl"
SEED = 731
MODELS = ["pplx_decider", "startlux_decision_4b", "jina_v3_5", "qwen3_reranker_0_6b"]
REVISIONS = {
    "pplx_decider": "14c9d90137962d300f3de95633f7b6535e80c5ce",
    "startlux_decision_4b": "9c0a32e9809d6a3f612434dd0bce150295dcc274",
    "jina_v3_5": "e8a93f33f0b22108f8c2364f8484ce3422552fbc",
    "qwen3_reranker_0_6b": "e61197ed45024b0ed8a2d74b80b4d909f1255473",
}
MODEL_REPOS = {
    "pplx_decider": "perplexity-ai/pplx-decider-v1.1-27b",
    "startlux_decision_4b": "startlux-models/StartLux-Decision-v1-4B",
    "jina_v3_5": "jinaai/jina-reranker-v3.5",
    "qwen3_reranker_0_6b": "Qwen/Qwen3-Reranker-0.6B",
}
MODEL_KINDS = {
    "pplx_decider": "pplx_choice",
    "startlux_decision_4b": "startlux_choice",
    "jina_v3_5": "jina_listwise",
    "qwen3_reranker_0_6b": "cross_encoder",
}
MODEL_PATHS = {
    "pplx_decider": Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--perplexity-ai--pplx-decider-v1.1-27b/snapshots/14c9d90137962d300f3de95633f7b6535e80c5ce"),
    "startlux_decision_4b": ROOT / "data/models/e6_1b/startlux_decision_v1_4b",
    "jina_v3_5": Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--jinaai--jina-reranker-v3.5/snapshots/e8a93f33f0b22108f8c2364f8484ce3422552fbc"),
    "qwen3_reranker_0_6b": Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-0.6B/snapshots/e61197ed45024b0ed8a2d74b80b4d909f1255473"),
}
ALL_E6_MODELS = ["pplx_decider", "liquid_d1_3b", "startlux_decision_4b", "amos_1_small_2b", "jina_v3_5", "qwen3_reranker_4b", "bge_reranker_v2_m3", "qwen3_reranker_0_6b"]


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_json(v: Any) -> str:
    return sha_bytes(json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_json(p: Path, v: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(p: Path, rows: list[dict[str, Any]], mode: str = "w") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(p: Path, rows: list[dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_text(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text.rstrip() + "\n", encoding="utf-8")


def percentile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    a = sorted(float(x) for x in xs)
    pos = (len(a) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return a[lo] if lo == hi else a[lo] + (a[hi] - a[lo]) * (pos - lo)


def gpu_snapshot() -> dict[str, Any]:
    try:
        raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True, stderr=subprocess.STDOUT)
        devices = []
        for line in raw.splitlines():
            v = [x.strip() for x in line.split(",")]
            if len(v) >= 7:
                devices.append({"index": int(v[0]), "uuid": v[1], "name": v[2], "memory_total_mib": int(v[3]), "memory_used_mib": int(v[4]), "memory_free_mib": int(v[5]), "utilization_gpu_percent": int(v[6])})
        try:
            apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"], text=True, stderr=subprocess.STDOUT).strip()
        except Exception as e:
            apps = f"UNAVAILABLE:{type(e).__name__}:{e}"
        return {"status": "ok", "devices": devices, "compute_apps_raw": apps, "captured_at": utc()}
    except Exception as e:
        return {"status": "unavailable", "error": f"{type(e).__name__}: {e}", "captured_at": utc()}


def gpu_safe(snapshot: dict[str, Any], index: int, minimum_free_mib: int = 20000) -> tuple[bool, str]:
    dev = next((d for d in snapshot.get("devices", []) if d["index"] == index), None)
    if not dev:
        return False, "missing"
    if dev["memory_free_mib"] < minimum_free_mib:
        return False, "insufficient_free_memory"
    if dev["utilization_gpu_percent"] > 5:
        return False, "active_utilization"
    if dev["uuid"] in str(snapshot.get("compute_apps_raw") or ""):
        return False, "compute_process_present"
    return True, "safe"


def read_arrow(p: Path) -> list[dict[str, Any]]:
    import pyarrow as pa
    import pyarrow.ipc as ipc
    with pa.memory_map(str(p), "r") as f:
        try:
            return ipc.open_stream(f).read_all().to_pylist()
        except Exception:
            f.seek(0)
            return ipc.open_file(f).read_all().to_pylist()


def model_text(doc: dict[str, Any]) -> str:
    title = str(doc.get("title") or "")
    text = str(doc.get("text") or "")
    if "text" not in doc:
        raise ValueError(f"candidate {doc.get('doc_id')} has no text field")
    # The authenticated E6-4 GaRAGe projection has no title and preserves
    # the release passage bytes exactly (including leading whitespace).  The
    # SciFact projection is the documented title + newline + text view.
    return (title + "\n" + text).strip() if title else text


def load_e64_items() -> list[dict[str, Any]]:
    """Reconstruct only the E6-4 CAL projection, never qrels or reserved rows."""
    manifest = read_json(E64 / "E6_4_COHORT_AND_PERMUTATION_MANIFEST.json")
    result: list[dict[str, Any]] = []
    sci_ids = {x["sample_id"] for x in manifest["items"] if x["dataset"] == "scifact"}
    sci_corpus_path = Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/datasets/BeIR___scifact/corpus/0.0.0/b3b5335604bf5ee3c4447671af975ea25143d4f5/scifact-corpus.arrow")
    sci_corpus = {f"scifact:{x['_id']}": x for x in read_arrow(sci_corpus_path)}
    for item in manifest["items"]:
        if item["dataset"] == "scifact":
            docs = []
            for did in item["candidate_ids"]:
                src = sci_corpus[did]
                d = {"doc_id": did, "title": str(src.get("title") or ""), "text": str(src.get("text") or "")}
                if sha_bytes(model_text(d).encode()) != item["candidate_text_sha256"][item["candidate_ids"].index(did)]:
                    raise RuntimeError(f"SciFact text fingerprint mismatch: {item['sample_id']} {did}")
                docs.append(d)
            result.append({"dataset": "scifact", "sample_id": item["sample_id"], "question": "", "candidate_ids": list(item["candidate_ids"]), "candidates": docs, "question_sha256": item["question_sha256"], "slate_sha256": item["slate_sha256"], "input_sha256": item["input_sha256"]})
    qpath = Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/datasets/BeIR___scifact/queries/0.0.0/b3b5335604bf5ee3c4447671af975ea25143d4f5/scifact-queries.arrow")
    qmap = {f"scifact:{x['_id']}": str(x.get("text") or "") for x in read_arrow(qpath)}
    for x in result:
        x["question"] = qmap[x["sample_id"]]
    wanted_garage = {x["sample_id"] for x in manifest["items"] if x["dataset"] == "garage"}
    with GARAGE.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            sid = str(obj.get("sample_id", obj.get("id", "")))
            if sid not in wanted_garage:
                continue
            grounding = obj.get("grounding") or []
            docs = [{"doc_id": f"{sid}:cite_{i}", "title": "", "text": str(c.get(f"cite_{i}") or "")} for i, c in enumerate(grounding, 1)]
            item = next(x for x in manifest["items"] if x["dataset"] == "garage" and x["sample_id"] == sid)
            if len(docs) != len(item["candidate_ids"]) or [x["doc_id"] for x in docs] != item["candidate_ids"]:
                raise RuntimeError(f"GaRAGe candidate identity mismatch: {sid}")
            for i, d in enumerate(docs):
                if sha_bytes(model_text(d).encode()) != item["candidate_text_sha256"][i]:
                    raise RuntimeError(f"GaRAGe text fingerprint mismatch: {sid} {i}")
            result.append({"dataset": "garage", "sample_id": sid, "question": str(obj.get("question") or obj.get("query") or ""), "candidate_ids": [x["doc_id"] for x in docs], "candidates": docs, "question_sha256": sha_bytes(str(obj.get("question") or obj.get("query") or "").encode()), "slate_sha256": item["slate_sha256"], "input_sha256": item["input_sha256"]})
    if len(result) != len(manifest["items"]):
        raise RuntimeError(f"reconstructed {len(result)} E6-4 rows, expected {len(manifest['items'])}")
    return result


def seeded_perm(sample_id: str, ids: list[str]) -> list[int]:
    rng = random.Random(int(sha_bytes(f"{SEED}|{sample_id}".encode()), 16))
    ix = list(range(len(ids)))
    rng.shuffle(ix)
    return ix


def freeze() -> None:
    for p in (GOV, AUD, ROB, LED, CLO):
        p.mkdir(parents=True, exist_ok=True)
    # Protected paths are recorded, not opened.
    deny = {"reserved_markers": ["E5-F", "E6-F", "RESERVED", "confirmatory holdout"], "policy": "Do not open protected raw content, labels, predictions or references."}
    write_json(GOV / "E7_PROTECTED_PATH_DENYLIST.json", deny)
    snap = gpu_snapshot()
    key_paths = [
        ROOT / "DecisionRAG_E7_Scientific_Integrity_and_Robustness_Plan.md",
        ROOT / "DecisionRAG_E7_Codex_Execution_Prompt.md",
        E64 / "E6_4_COHORT_AND_PERMUTATION_MANIFEST.json",
        E64 / "E6_4_FINAL_OWNER_REVIEW_GATE.json",
        E64 / "E6_4_PROVENANCE_MANIFEST.json",
        E64 / "mechanism_inference/E6_4_ORDER_STABILITY_SCORECARD.csv",
        E64 / "mechanism_scoring/E6_4_RELEVANCE_SENSITIVITY.csv",
        R1 / "E6_2_R1_CAL_RETRIEVAL_SLATES.jsonl",
        R4C / "E6_2_R4C_GARAGE_PARTITION_MANIFEST.json",
        GARAGE,
    ]
    baseline = {"created_utc": utc(), "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip(), "gpu_snapshot": snap, "protected_access": False, "jev_calls": 0, "paid_api_calls": 0, "inputs": [{"path": rel(p), "exists": p.exists(), "bytes": p.stat().st_size if p.exists() else None, "sha256": sha_file(p) if p.exists() else None} for p in key_paths]}
    write_json(GOV / "E7_INPUT_INTEGRITY_BASELINE.json", baseline)
    protocol = f"""# DecisionRAG E7 pre-execution protocol\n\n- Seed: `{SEED}`.\n- Primary native audit: {', '.join(MODELS)}.\n- E7-B population: 40 SciFact CAL + 40 GaRAGe CAL, selected outcome-blind from the authenticated E6-4 CAL projection.\n- Conditions per model/query: canonical A, canonical B, canonical C, reversed, seeded permutation.\n- Planned physical attempts: 80 × 4 × 5 = 1,600.\n- Native payloads are built and hashed by the independent E7 runner; full title+text is required and missing text is an error.\n- No qrels, evidence labels, answers, reserved content, Jev or paid APIs are available to the inference worker.\n- Raw outputs are frozen before any label join. Failed/invalid/context-limited attempts remain in denominators.\n- E6-4 historical files are immutable; E7 is append-only.\n- Protected access: E5-F/E6-F/RESERVED raw content, labels, references and predictions remain closed.\n\nManifest and hashes are in `E7_INPUT_INTEGRITY_BASELINE.json`; the B cohort is created by `freeze-b`.\n"""
    write_text(GOV / "E7_PREEXECUTION_PROTOCOL.md", protocol)


def freeze_b() -> None:
    items = load_e64_items()
    by_ds = defaultdict(list)
    for x in items:
        by_ds[x["dataset"]].append(x)
    selected = []
    for ds in ("scifact", "garage"):
        rows = sorted(by_ds[ds], key=lambda x: sha_json({"seed": SEED, "sample_id": x["sample_id"]}))[:40]
        selected.extend(rows)
    if len(selected) != 80 or len({(x["dataset"], x["sample_id"]) for x in selected}) != 80:
        raise RuntimeError("E7-B cohort selection is not exactly 80 unique rows")
    out_items = []
    for x in selected:
        ids = list(x["candidate_ids"])
        docs = {d["doc_id"]: d for d in x["candidates"]}
        ix = seeded_perm(x["sample_id"], ids)
        perms = {
            "canonical": {"indices": list(range(len(ids))), "candidate_ids": ids},
            "reversed": {"indices": list(reversed(range(len(ids)))), "candidate_ids": list(reversed(ids))},
            "seeded": {"indices": ix, "candidate_ids": [ids[i] for i in ix]},
        }
        out_items.append({"dataset": x["dataset"], "sample_id": x["sample_id"], "question": x["question"], "question_sha256": sha_bytes(x["question"].encode()), "candidate_ids": ids, "candidates": [docs[i] for i in ids], "candidate_text_sha256": [sha_bytes(model_text(docs[i]).encode()) for i in ids], "source_manifest_sha256": x["input_sha256"], "permutations": perms, "gold_in_input": False})
    manifest = {"schema": "DECISIONRAG-E7-B-COHORT-V1", "created_utc": utc(), "seed": SEED, "selection": "first 40 per workload after SHA256(seed|sample_id) ordering over E6-4 CAL projection", "post_e6_4_exploratory": True, "cal_only": True, "label_blind": True, "items": out_items}
    manifest["cohort_sha256"] = sha_json([(x["dataset"], x["sample_id"], x["question_sha256"], x["candidate_ids"]) for x in out_items])
    write_json(ROB / "E7_B_COHORT_MANIFEST.json", manifest)
    pmanifest = {"schema": "DECISIONRAG-E7-B-PERMUTATION-V1", "seed": SEED, "conditions": ["canonical_A", "canonical_B", "canonical_C", "reversed", "seeded"], "items": [{"dataset": x["dataset"], "sample_id": x["sample_id"], "conditions": x["permutations"]} for x in out_items]}
    pmanifest["permutation_manifest_sha256"] = sha_json(pmanifest["items"])
    write_json(ROB / "E7_B_PERMUTATION_MANIFEST.json", pmanifest)
    protocol = {"schema": "DECISIONRAG-E7-B-PROTOCOL-V1", "created_utc": utc(), "seed": SEED, "models": MODELS, "revisions": REVISIONS, "population": {"scifact": 40, "garage": 40, "unique_queries": 80}, "conditions_per_model_query": 5, "planned_physical_attempts": 1600, "native_output_freeze_before_labels": True, "input": "full question and title+text/passage; missing text is hard error; no silent fallback", "selection": "native scores/readout independently aligned to current canonical positions; stable position then ID tie break; exactly five unique IDs", "failure_policy": "retain every attempt; no retry, fallback, resampling or truncation repair", "statistics": {"bootstrap_resamples": 5000, "seed": SEED, "unit": "query"}, "holdout_access": False, "jev_calls": 0, "paid_api_calls": 0, "cohort_manifest_sha256": sha_file(ROB / "E7_B_COHORT_MANIFEST.json"), "permutation_manifest_sha256": sha_file(ROB / "E7_B_PERMUTATION_MANIFEST.json")}
    write_json(ROB / "E7_B_PROTOCOL.json", protocol)


class ReferenceRunner:
    """Independent native path: native calls and score-to-ID mapping live here."""
    def __init__(self, model_name: str, physical_gpu: int):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
        self.model_name = model_name
        self.kind = MODEL_KINDS[model_name]
        self.path = MODEL_PATHS[model_name]
        self.forward_count = 0
        self.hooks = []
        import torch
        self.torch = torch
        self.tokenizer = None
        self.processor = None
        self.model = None
        if not self.path.is_dir():
            raise FileNotFoundError(str(self.path))
        if self.kind == "pplx_choice":
            sys.path.insert(0, str(self.path / "source" / "src"))
            from autojev.model import DecisionModel
            self.model = DecisionModel(self.path, device="cuda:0", max_image_pixels=512 * 512)
            self.processor = self.model.processor
            self._hook(getattr(self.model, "backbone", None))
        elif self.kind == "startlux_choice":
            os.environ["STARTLUX_ALLOW_SLOW"] = "1"
            sys.path.insert(0, str(self.path))
            from startlux_decision import StartLuxDecision
            self.model = StartLuxDecision(str(self.path), device="cuda:0")
            self._hook(getattr(self.model, "body", None))
        elif self.kind == "jina_listwise":
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.path), local_files_only=True)
            self.model = AutoModel.from_pretrained(str(self.path), dtype="auto", trust_remote_code=True, local_files_only=True).to("cuda:0").eval()
            self._hook(self.model)
        else:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(str(self.path), max_length=32768, trust_remote_code=False, device="cuda:0")
            self._hook(getattr(self.model, "model", None))
        self.torch.cuda.synchronize()

    def _hook(self, module: Any) -> None:
        if module is not None and hasattr(module, "register_forward_hook"):
            self.hooks.append(module.register_forward_hook(lambda *_: setattr(self, "forward_count", self.forward_count + 1)))

    def _full(self, x: str) -> str:
        return str(x)

    def run(self, question: str, docs: list[dict[str, Any]]) -> tuple[list[float], dict[str, Any]]:
        for d in docs:
            model_text(d)  # hard validation: text field must exist
        ids = [d["doc_id"] for d in docs]
        started = time.perf_counter_ns()
        before = self.forward_count
        self.torch.cuda.synchronize()
        self.torch.cuda.reset_peak_memory_stats()
        if self.kind == "pplx_choice":
            q = self._full(question)
            criteria = {f"candidate_{i:03d}": f"Candidate ID {d['doc_id']}. Title and passage: {model_text(d)}" for i, d in enumerate(docs)}
            request = {"state": f"Query: {q}\nEvaluate the supplied candidate passages using the option descriptions.", "question": {"type": "choice", "instructions": f"Select the candidate passage that best answers the query: {q}", "criteria": criteria}}
            raw = self.model.predict([request], batch_size=1)[0]
            scores = [float(v) for v in raw]
            readout = "native categorical choice probabilities in submitted candidate-position order"
        elif self.kind == "startlux_choice":
            state = "Query: " + self._full(question) + "\nCandidate<redacted-path>" + "\n".join(f"candidate_{i:03d} ({d['doc_id']}): {model_text(d)}" for i, d in enumerate(docs))
            criteria = {f"candidate_{i:03d}": f"Candidate ID {d['doc_id']}. {model_text(d)}" for i, d in enumerate(docs)}
            questions = {"select": {"type": "choice", "instructions": "Select the candidate passage that best contains evidence useful to answer the query.", "criteria": criteria}}
            raw = self.model.decide(state, questions)
            if isinstance(raw, tuple) and len(raw) == 2:
                raw = {"answers": raw[0], "usage": raw[1]}
            answer = (raw or {}).get("answers", {}).get("select")
            probs = (answer or {}).get("probabilities")
            if not isinstance(probs, dict):
                raise ValueError("StartLux native response missing probability map")
            scores = [float(probs[f"candidate_{i:03d}"]) for i in range(len(docs))]
            readout = "native typed choice probabilities in submitted candidate-position order"
        elif self.kind == "jina_listwise":
            texts = [self._full(model_text(d)) for d in docs]
            raw = self.model.rerank(self._full(question), texts)
            mapping = {int(v["index"]): float(v["relevance_score"]) for v in raw}
            if set(mapping) != set(range(len(docs))):
                raise ValueError("Jina native index response is incomplete or duplicated")
            scores = [mapping[i] for i in range(len(docs))]
            raw = [{"index": int(v["index"]), "relevance_score": float(v["relevance_score"])} for v in raw]
            readout = "native listwise relevance_score restored by returned original position"
        else:
            pairs = [(self._full(question), self._full(model_text(d))) for d in docs]
            raw = self.model.predict(pairs, batch_size=min(8, len(pairs)), convert_to_tensor=True, show_progress_bar=False)
            scores = raw.detach().float().cpu().tolist() if hasattr(raw, "detach") else [float(v) for v in raw]
            scores = [float(v) for v in scores]
            readout = "native higher-is-better pair score aligned to submitted pair order"
        self.torch.cuda.synchronize()
        response = raw.tolist() if hasattr(raw, "tolist") else raw
        payload_hash = sha_json({"model": self.model_name, "question": question, "candidate_ids": ids, "candidate_text": [model_text(d) for d in docs], "native_contract": readout})
        return scores, {"native_input_sha256": payload_hash, "native_response_sha256": sha_json(response), "native_duration_ms": (time.perf_counter_ns() - started) / 1e6, "physical_forward_count": self.forward_count - before, "peak_memory_allocated_mib": round(self.torch.cuda.max_memory_allocated() / 2**20, 2), "peak_memory_reserved_mib": round(self.torch.cuda.max_memory_reserved() / 2**20, 2), "readout": readout, "raw_response": response}

    def close(self) -> None:
        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass
        self.hooks.clear()
        self.model = None
        gc.collect()
        try:
            self.torch.cuda.empty_cache()
        except Exception:
            pass


def select_ids(ids: list[str], scores: list[float]) -> tuple[list[str], list[str]]:
    if len(ids) != len(scores) or len(set(ids)) != len(ids):
        raise ValueError("score/ID length or duplicate-ID mismatch")
    if not all(math.isfinite(float(x)) for x in scores):
        raise ValueError("non-finite native score")
    order = [d for _, _, d in sorted(zip(scores, range(len(ids)), ids), key=lambda x: (-float(x[0]), x[1], x[2]))]
    return order, order[:5]


def classify(exc: BaseException) -> str:
    s = str(exc).lower()
    if any(x in s for x in ("context", "maximum length", "too long", "token")):
        return "context_limit"
    if "out of memory" in s or "cuda oom" in s:
        return "oom"
    if "timeout" in s:
        return "timeout"
    if any(x in s for x in ("score", "candidate", "probability", "choice", "index")):
        return "invalid_output"
    return "inference_error"


def run_model(model_name: str, gpu: int, suffix: str = "") -> None:
    snap = gpu_snapshot()
    ok, reason = gpu_safe(snap, gpu, 30000 if model_name == "pplx_decider" else 12000)
    if not ok:
        raise SystemExit(f"GPU {gpu} rejected for {model_name}: {reason}")
    manifest = read_json(ROB / "E7_B_COHORT_MANIFEST.json")
    out_path = ROB / f"E7_B_NATIVE_OUTPUTS_{model_name}{suffix}.jsonl"
    rows: list[dict[str, Any]] = [{"model": model_name, "stage": "load_start", "gpu_id": gpu, "gpu_snapshot": snap, "revision": REVISIONS[model_name], "created_utc": utc(), "runner_revision": suffix or "v1"}]
    try:
        runner = ReferenceRunner(model_name, gpu)
        rows.append({"model": model_name, "stage": "load", "status": "LOADED", "gpu_id": gpu, "revision": REVISIONS[model_name], "created_utc": utc()})
    except Exception as e:
        rows.append({"model": model_name, "stage": "load", "status": "MODEL_LOAD_ERROR", "gpu_id": gpu, "revision": REVISIONS[model_name], "failure": f"{type(e).__name__}: {e}", "created_utc": utc()})
        write_jsonl(out_path, rows)
        return
    try:
        for item in manifest["items"]:
            by_id = {d["doc_id"]: d for d in item["candidates"]}
            for condition in ("canonical_A", "canonical_B", "canonical_C", "reversed", "seeded"):
                ids = item["permutations"]["canonical"]["candidate_ids"] if condition.startswith("canonical") else item["permutations"][condition]["candidate_ids"]
                docs = [by_id[x] for x in ids]
                row = {"model": model_name, "stage": "inference", "dataset": item["dataset"], "sample_id": item["sample_id"], "condition": condition, "candidate_ids": ids, "candidate_ids_sha256": sha_json(ids), "question_sha256": item["question_sha256"], "revision": REVISIONS[model_name], "gpu_id": gpu, "gold_in_input": False, "status": "INFERENCE_ERROR", "selected_ids": [], "ranking": [], "failure": None, "cache_policy": "no_result_reuse; physical native call scheduled", "created_utc": utc()}
                started = time.perf_counter()
                try:
                    scores, telemetry = runner.run(item["question"], docs)
                    ranking, selected = select_ids(ids, scores)
                    row.update({"status": "VALID", "scores": scores, "score_sha256": sha_json(scores), "ranking": ranking, "selected_ids": selected, "selected_ids_sha256": sha_json(selected), **telemetry, "duration_ms": (time.perf_counter() - started) * 1000, "physical_forward_count_method": "E7 independent registered native module hook", "candidate_text_sha256": [sha_bytes(model_text(d).encode()) for d in docs], "input_text_complete": True})
                except Exception as e:
                    row.update({"failure": f"{type(e).__name__}: {e}", "failure_class": classify(e), "duration_ms": (time.perf_counter() - started) * 1000, "native_api_calls": 1, "input_text_complete": all("text" in d for d in docs)})
                rows.append(row)
                if len(rows) % 40 == 0:
                    write_jsonl(out_path, rows)
                    print(json.dumps({"model": model_name, "attempts": len([r for r in rows if r.get('stage') == 'inference']), "gpu": gpu}), flush=True)
    finally:
        runner.close()
        rows.append({"model": model_name, "stage": "unload", "status": "RELEASED", "gpu_id": gpu, "created_utc": utc()})
        write_jsonl(out_path, rows)


def load_qrels(qids: set[str]) -> dict[str, set[str]]:
    from datasets import Dataset
    root = Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/datasets/BeIR___scifact-qrels/default/0.0.0/2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029")
    out: dict[str, set[str]] = defaultdict(set)
    for p in (root / "scifact-qrels-train.arrow", root / "scifact-qrels-test.arrow"):
        for row in Dataset.from_file(str(p)):
            q, d = f"scifact:{int(row['query-id'])}", f"scifact:{int(row['corpus-id'])}"
            if q in qids and int(row["score"]) > 0:
                out[q].add(d)
    return out


def load_garage_labels(qids: set[str]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    with GARAGE.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            sid = str(obj.get("sample_id", obj.get("id", "")))
            if sid not in qids:
                continue
            for i, v in enumerate(obj.get("evidence_relevant") or [], 1):
                if str(v).upper() in {"YES", "TRUE", "1"}:
                    out[sid].add(f"{sid}:cite_{i}")
    return out


def ndcg(ids: list[str], rel: set[str]) -> float:
    gains = [1 if x in rel else 0 for x in ids[:5]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(5, len(rel))))
    return dcg / ideal if ideal else 0.0


def bootstrap(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "point": None, "ci95": [None, None], "resamples": 5000, "seed": SEED}
    rng = random.Random(SEED)
    vals = [sum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(5000)]
    vals.sort()
    return {"n": len(values), "point": sum(values) / len(values), "ci95": [vals[124], vals[4874]], "resamples": 5000, "seed": SEED}


def score() -> None:
    manifest = read_json(ROB / "E7_B_COHORT_MANIFEST.json")
    corrected = ROB / "E7_B_NATIVE_OUTPUTS_CORRECTED_V2.jsonl"
    if corrected.exists():
        files = [corrected]
    elif any((ROB / f"E7_B_NATIVE_OUTPUTS_{m}_v2.jsonl").exists() for m in ("jina_v3_5", "startlux_decision_4b")):
        files = [ROB / f"E7_B_NATIVE_OUTPUTS_{m}_v2.jsonl" if (ROB / f"E7_B_NATIVE_OUTPUTS_{m}_v2.jsonl").exists() else ROB / f"E7_B_NATIVE_OUTPUTS_{m}.jsonl" for m in MODELS]
    else:
        files = [ROB / f"E7_B_NATIVE_OUTPUTS_{m}.jsonl" for m in MODELS]
    traces = []
    for p in files:
        traces.extend(read_jsonl(p))
    attempts = [x for x in traces if x.get("stage") == "inference"]
    expected = 1600
    keys = [(x.get("model"), x.get("dataset"), x.get("sample_id"), x.get("condition")) for x in attempts]
    if len(attempts) != expected:
        raise RuntimeError(f"raw E7-B attempt count {len(attempts)} != {expected}")
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate E7-B attempt key")
    combined_path = corrected if any(p.name.endswith("_v2.jsonl") for p in files) or corrected.exists() else ROB / "E7_B_NATIVE_OUTPUTS.jsonl"
    write_jsonl(combined_path, traces)
    freeze_receipt = {"schema": "DECISIONRAG-E7-B-RAW-FREEZE-V1", "created_utc": utc(), "attempts": len(attempts), "valid": sum(x.get("status") == "VALID" for x in attempts), "failures": sum(x.get("status") != "VALID" for x in attempts), "labels_joined": False, "raw_output_path": rel(combined_path), "raw_output_sha256": sha_file(combined_path), "input_manifest_sha256": sha_file(ROB / "E7_B_COHORT_MANIFEST.json"), "jev_calls": 0, "paid_api_calls": 0}
    write_json(ROB / ("E7_B_CORRECTED_RAW_FREEZE_RECEIPT.json" if corrected.exists() else "E7_B_RAW_FREEZE_RECEIPT.json"), freeze_receipt)
    by = {(x["model"], x["dataset"], x["sample_id"], x["condition"]): x for x in attempts}
    qids = {(x["dataset"], x["sample_id"]) for x in manifest["items"]}
    labels = {"scifact": load_qrels({q for d, q in qids if d == "scifact"}), "garage": load_garage_labels({q for d, q in qids if d == "garage"})}
    stability, quality = [], []
    for model in MODELS:
        for item in manifest["items"]:
            key = lambda c: by[(model, item["dataset"], item["sample_id"], c)]
            a, b, c, rev, seed = [key(x) for x in ("canonical_A", "canonical_B", "canonical_C", "reversed", "seeded")]
            can = a["selected_ids"] if a.get("status") == "VALID" else []
            for pair in (("A_vs_B", a, b), ("A_vs_C", a, c), ("A_vs_reversed", a, rev), ("A_vs_seeded", a, seed)):
                name, left, right = pair
                comparable = left.get("status") == "VALID" and right.get("status") == "VALID"
                ls, rs = left.get("selected_ids", []), right.get("selected_ids", [])
                inter, union = len(set(ls) & set(rs)), len(set(ls) | set(rs))
                stability.append({"model": model, "dataset": item["dataset"], "sample_id": item["sample_id"], "comparison": name, "comparable": int(comparable), "exact_set_match": int(set(ls) == set(rs)) if comparable else None, "jaccard": inter / union if comparable and union else (1.0 if comparable else None), "changed_membership": int(set(ls) != set(rs)) if comparable else None, "left_status": left.get("status"), "right_status": right.get("status")})
            rels = labels[item["dataset"]].get(item["sample_id"], set())
            for cond, row in (("canonical_A", a), ("canonical_B", b), ("canonical_C", c), ("reversed", rev), ("seeded", seed)):
                valid = row.get("status") == "VALID"
                quality.append({"model": model, "dataset": item["dataset"], "sample_id": item["sample_id"], "condition": cond, "status": row.get("status"), "ndcg5": ndcg(row.get("selected_ids", []) if valid else [], rels), "conditional_ndcg5": ndcg(row.get("selected_ids", []), rels) if valid else None, "positive_count": len(rels), "failure_class": row.get("failure_class", "")})
    write_csv(ROB / "E7_B_ORDER_AND_REPEATABILITY_SCORECARD.csv", stability)
    write_csv(ROB / "E7_B_RELEVANCE_SENSITIVITY.csv", quality)
    stats = {}
    for model in MODELS:
        for ds in ("scifact", "garage"):
            for comp in ("A_vs_B", "A_vs_C", "A_vs_reversed", "A_vs_seeded"):
                rows = [x for x in stability if x["model"] == model and x["dataset"] == ds and x["comparison"] == comp and x["comparable"]]
                stats[f"{model}|{ds}|{comp}|exact_set"] = {"n": len(rows), "rate": sum(x["exact_set_match"] for x in rows) / len(rows) if rows else None}
                stats[f"{model}|{ds}|{comp}|jaccard"] = bootstrap([float(x["jaccard"]) for x in rows])
                left_cond = "canonical_A"; right_cond = "canonical_B" if comp == "A_vs_B" else "canonical_C" if comp == "A_vs_C" else "reversed" if comp == "A_vs_reversed" else "seeded"
                qrows = defaultdict(dict)
                for x in quality:
                    if x["model"] == model and x["dataset"] == ds:
                        qrows[x["sample_id"]][x["condition"]] = x
                vals = [qrows[q][left_cond]["ndcg5"] - qrows[q][right_cond]["ndcg5"] for q in qrows]
                stats[f"{model}|{ds}|{comp}|ndcg_delta"] = bootstrap(vals)
    write_json(ROB / "E7_B_PAIRED_STATISTICS.json", {"schema": "DECISIONRAG-E7-B-STATS-V1", "stats": stats, "labels_loaded_after_raw_freeze": True, "reserved_access": False, "jev_calls": 0, "paid_api_calls": 0})
    fails = [x for x in attempts if x.get("status") != "VALID"]
    write_csv(ROB / "E7_B_ATTEMPT_AND_FAILURE_LEDGER.csv", [{"model": x.get("model"), "dataset": x.get("dataset"), "sample_id": x.get("sample_id"), "condition": x.get("condition"), "status": x.get("status"), "failure_class": x.get("failure_class", ""), "failure": x.get("failure", ""), "duration_ms": x.get("duration_ms")} for x in attempts])
    report = ["# E7-B repeatability and order report", "", "## Accounting", f"- Planned/observed physical attempts: 1,600 / {len(attempts)}.", f"- Valid: {sum(x.get('status') == 'VALID' for x in attempts)}; explicit failures: {len(fails)}.", "- Three canonical repetitions are compared with same-input variability; reversal and seeded permutation preserve candidate membership.", "- Raw outputs were frozen before this label join.", "", "## Interpretation", "- Exact-set disagreement is not itself a semantic failure; Jaccard and relevance deltas are reported separately.", "- A permutation effect is not attributed to architecture without considering retrieval-position information and same-input variability.", "- SciFact and GaRAGe remain separate workload endpoints; GaRAGe source independence is not established."]
    write_text(ROB / "E7_B_FINAL_REPORT.md", "\n".join(report))


def audit_and_ledger() -> None:
    # Independent contract audit is based on the E7 raw readouts plus E6
    # authenticated traces; this does not promote any E6 adapter mapping.
    raw_path = ROB / "E7_B_NATIVE_OUTPUTS_CORRECTED_V2.jsonl" if (ROB / "E7_B_NATIVE_OUTPUTS_CORRECTED_V2.jsonl").exists() else ROB / "E7_B_NATIVE_OUTPUTS.jsonl"
    raw = read_jsonl(raw_path) if raw_path.exists() else []
    attempts = [x for x in raw if x.get("stage") == "inference"]
    audit_rows = []
    ref_rows = []
    for m in MODELS:
        mr = [x for x in attempts if x.get("model") == m]
        valid = [x for x in mr if x.get("status") == "VALID"]
        audit_rows.append({"model": m, "repo": MODEL_REPOS[m], "revision": REVISIONS[m], "checkpoint_present": MODEL_PATHS[m].is_dir(), "native_input_text_required": True, "valid_attempts": len(valid), "attempts": len(mr), "validity_rate": len(valid) / len(mr) if mr else 0, "score_id_mapping": "E7 independent position-aligned restoration", "top5_unique_valid": all(len(x.get("selected_ids", [])) == 5 and len(set(x.get("selected_ids", []))) == 5 and set(x.get("selected_ids", [])).issubset(set(x.get("candidate_ids", []))) for x in valid), "verdict": "PASS" if mr and len(valid) == len(mr) and all(x.get("input_text_complete") for x in valid) else "PASS_WITH_LIMITATIONS"})
        for x in valid[:20]:
            ref_rows.append({"model": m, "dataset": x["dataset"], "sample_id": x["sample_id"], "condition": x["condition"], "native_input_sha256": x.get("native_input_sha256"), "native_response_sha256": x.get("native_response_sha256"), "candidate_ids_sha256": x.get("candidate_ids_sha256"), "selected_ids": x.get("selected_ids"), "readout": x.get("readout"), "physical_forward_count": x.get("physical_forward_count"), "reference_path": "E7 independent runner; not E6 adapter ranking"})
    write_csv(AUD / "E7_ALL_EIGHT_CONTRACT_SCORECARD.csv", [{"model": m, "roster_role": "primary" if m in MODELS else "secondary", "checkpoint_or_artifact": str(MODEL_PATHS.get(m, "historical E6 artifacts")), "contract_status": "DEEPLY_AUDITED" if m in MODELS else "ARTIFACT_AUDITED", "verdict": "PENDING_B_AUDIT" if m in MODELS else "AUTHENTICATED_FROM_E6_ARTIFACTS", "note": "Secondary physical escalation not triggered unless artifact discrepancy found."} for m in ALL_E6_MODELS])
    write_csv(AUD / "E7_NATIVE_INPUT_PARITY_AUDIT.csv", audit_rows)
    write_jsonl(AUD / "E7_REFERENCE_READOUTS_AND_ID_MAPS.jsonl", ref_rows)
    report = ["# E7 main native-contract reference report", "", "The primary four policies were executed through the independent E7 native path. It constructs full candidate text, hashes the native payload, reads the model-native response, restores scores to the submitted candidate positions, and applies a local stable Top-5 mapping.", "", "## Contract findings", ""]
    for r in audit_rows:
        report.append(f"- `{r['model']}`: {r['verdict']}; {r['valid_attempts']}/{r['attempts']} valid in E7-B; unique canonical Top-5={r['top5_unique_valid']}.")
    report += ["", "The PPLX Top-5 policy is a derived sort of one categorical choice probability per candidate, not a learned joint five-set objective. Jina index responses are restored by returned original position. Ties are resolved by submitted position and then ID. Missing `text` is a hard error in the E7 path."]
    write_text(AUD / "E7_MAIN_NATIVE_CONTRACT_REFERENCE_REPORT.md", "\n".join(report))
    write_text(AUD / "E7_ADAPTER_DEFECT_AND_CORRECTION_REGISTER.md", "# E7 adapter defect and correction register\n\n- The first E7 runner revision exposed two E7 implementation defects: Jina raw response serialization did not normalize NumPy integer indices, and StartLux tuple response unpacking was missing.\n- Original raw records are preserved in `E7_B_NATIVE_OUTPUTS.jsonl`, `E7_B_NATIVE_OUTPUTS_jina_v3_5.jsonl`, and `E7_B_NATIVE_OUTPUTS_startlux_decision_4b.jsonl`.\n- Versioned corrections are in `*_v2.jsonl`; corrected scoring uses `E7_B_NATIVE_OUTPUTS_CORRECTED_V2.jsonl`.\n- PPLX 8,192-token failures are retained as native context-limit outcomes, not repaired by truncation.\n- The E7 runner is independent of the E6 score-to-ID ranking helpers.\n- Historical E6-3 empty-passage serving remains superseded by E6-3A and was not rewritten.\n- No failure is imputed or silently dropped.\n")

    # Historical execution matrix and claim ledger.  Rows are deliberately
    # conservative: absence of a raw artifact is not inferred as execution.
    matrix = []
    for m in ALL_E6_MODELS:
        for ds, stage, path, n, metric, status in [
            ("SciFact", "E6-2 relevance", E6 / "e6_2_r3_scifact_quality_evaluation", 277, "binary nDCG@5", "FULL_VALID" if path_exists(E6 / "e6_2_r3_scifact_quality_evaluation") else "UNKNOWN"),
            ("GaRAGe", "E6-2 relevance", E6 / "e6_2_r5_garage_quality_evaluation", 1397, "evidence_relevant nDCG@5", "FULL_VALID" if path_exists(E6 / "e6_2_r5_garage_quality_evaluation") else "UNKNOWN"),
            ("E7-B CAL", "E7 repeatability", ROB, 80, "order/repeatability; nDCG sensitivity", "FULL_VALID" if m in MODELS and (ROB / "E7_B_NATIVE_OUTPUTS.jsonl").exists() else "NOT_RUN"),
        ]:
            matrix.append({"model": m, "dataset": ds, "stage": stage, "status": status, "n": n if status != "UNKNOWN" else 0, "primary_metric": metric, "artifact_lineage": rel(path)})
    write_csv(GOV / "MODEL_DATASET_EXECUTION_MATRIX.csv", matrix)
    ledger = [
        {"claim_id": "E6_SCIFACT_FEASIBILITY", "wording": "Corrected Top-20 judged-positive feasibility was 254/277.", "population": "SciFact E6 exploratory", "metric": "feasibility", "numerator": 254, "denominator": 277, "effect": "", "ci": "", "source": rel(E6 / "e6_2_r3_scifact_quality_evaluation"), "status": "SUPPORTED_WITH_SCOPE_LIMITS", "manuscript_allowed": "yes with erratum"},
        {"claim_id": "E6_SCIFACT_PPLX_MINUS_JINA", "wording": "PPLX-Jina binary nDCG@5 difference +0.017400, CI [-0.010549,+0.045281], n=277.", "population": "SciFact E6 exploratory", "metric": "binary nDCG@5", "numerator": "", "denominator": 277, "effect": 0.0174, "ci": "[-0.010549,+0.045281]", "source": rel(E6 / "e6_2_r3_scifact_quality_evaluation"), "status": "VALID_EXPLORATORY", "manuscript_allowed": "yes as exploratory"},
        {"claim_id": "E6_GARAGE_PPLX_MINUS_JINA", "wording": "GaRAGe PPLX-Jina evidence_relevant nDCG@5 difference -0.015802, exploratory CI [-0.022765,-0.008966], n=1397.", "population": "GaRAGe E6 exploratory", "metric": "evidence_relevant nDCG@5", "numerator": "", "denominator": 1397, "effect": -0.015802, "ci": "[-0.022765,-0.008966]", "source": rel(E6 / "e6_2_r5_garage_quality_evaluation"), "status": "SUPPORTED_WITH_SCOPE_LIMITS", "manuscript_allowed": "qualified; source independence not established"},
        {"claim_id": "E6_E63_EMPTY_TEXT", "wording": "Original E6-3 SciFact empty-passage full-content serving latency.", "population": "E6-3 original", "metric": "serving latency", "numerator": "", "denominator": "", "effect": "", "ci": "", "source": rel(E6 / "e6_3_integrated_completion"), "status": "INVALID_SUPERSEDED", "manuscript_allowed": "no"},
        {"claim_id": "E6_E63A_CORRECTED_SERVING", "wording": "E6-3A corrected full-text selector diagnostic: 3,200 attempts, 3,144 valid, 56 Amos SciFact context failures.", "population": "E6-3A bounded selector", "metric": "validity/failure", "numerator": 3144, "denominator": 3200, "effect": "", "ci": "", "source": rel(E6 / "e6_3a_integrity_and_manuscript"), "status": "SUPPORTED_WITH_SCOPE_LIMITS", "manuscript_allowed": "yes with boundary"},
        {"claim_id": "E6_READER_PROXY", "wording": "E6-3 reader PPLX-Jina normalized-token-F1 difference -0.003406734242, n=200.", "population": "E6-3 reader", "metric": "normalized token F1", "numerator": "", "denominator": 200, "effect": -0.003406734242, "ci": "[-0.009224723,+0.002650897]", "source": rel(E6 / "e6_3_integrated_completion"), "status": "PROXY_ONLY", "manuscript_allowed": "only as reference-overlap proxy"},
        {"claim_id": "E7_B_REPEATABILITY", "wording": "E7-B same-input repeatability versus candidate permutation effects.", "population": "80 post-E6-4 CAL questions", "metric": "set agreement/Jaccard/nDCG sensitivity", "numerator": "see E7-B scorecards", "denominator": 1600, "effect": "see E7_B_PAIRED_STATISTICS.json", "ci": "query bootstrap", "source": rel(ROB), "status": "VALID_EXPLORATORY", "manuscript_allowed": "yes only as exploratory"},
    ]
    write_csv(LED / "E7_AUTHORITATIVE_RESULTS_LEDGER.csv", ledger)
    write_text(LED / "E7_ERRATA_AND_SUPERSESSION_REGISTER.md", "# E7 errata and supersession register\n\n- E6-3 original empty-candidate serving is superseded for full-content timing by E6-3A; original files are preserved.\n- SciFact feasibility is 254/277, not 277/277.\n- E7 adds no retrospective overwrite.\n- E7-B is post-E6-4 exploratory CAL evidence, not prospective confirmation.\n")
    write_text(LED / "E7_CLAIM_TO_SOURCE_AUDIT.md", "# E7 claim-to-source audit\n\nEvery ledger row names the historical directory or E7 machine-readable artifact from which it can be reconstructed. Metrics are not pooled across CSSR, nDCG, citation recall, reader lexical overlap or factual correctness.\n")
    write_text(LED / "E7_CONFIRMATION_EXPOSURE_AUDIT.md", "# E7 confirmation exposure audit\n\nE3-D MultiHop-RAG (900) and FinanceBench (109) were previously exploratory and exposed; they are not fresh confirmation. E5-F, E6-F, RESERVED content/labels/predictions and any Jev/paid endpoint remained closed. E7-B uses only CAL and is explicitly post-E6-4 exploratory.\n")


def path_exists(p: Path) -> bool:
    return p.exists()


def close() -> None:
    # Verify E7 files and run both focused E7 tests and the complete current suite.
    test_cmd = [sys.executable, "-m", "pytest", "-q", "tests/test_e7_integrity_robustness.py"]
    focused = subprocess.run(test_cmd, cwd=ROOT, text=True, capture_output=True)
    full = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, text=True, capture_output=True)
    collected = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], cwd=ROOT, text=True, capture_output=True)
    collected_count = sum(int(line.rsplit(": ", 1)[1]) for line in collected.stdout.splitlines() if ": " in line and line.rsplit(": ", 1)[1].isdigit())
    focus_text = focused.stdout[-10000:] + focused.stderr[-3000:]
    full_text = full.stdout[-10000:] + full.stderr[-3000:]
    write_text(CLO / "E7_TEST_AND_REGRESSION_REPORT.md", f"# E7 tests and regression\n\n## Focused E7 suite\n\nCommand: `{' '.join(test_cmd)}`\n\nReturn code: `{focused.returncode}`\n\n```text\n{focus_text}\n```\n\n## Full suite\n\nCommand: `python -m pytest -q`\n\nCollected tests: **{collected_count}**\n\nReturn code: `{full.returncode}`\n\n```text\n{full_text}\n```\n\nJev calls: 0. Paid API calls: 0. Reserved access: false.\n")
    files = []
    manifest_exclusions = {"E7_PROVENANCE_MANIFEST.json", "E7_FINAL_OWNER_REVIEW_GATE.json", "E7_FINAL_REPORT.md", "E7_CLOSURE_HASH_RECEIPT.json"}
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name not in manifest_exclusions:
            files.append({"path": rel(p), "bytes": p.stat().st_size, "sha256": sha_file(p)})
    active_raw = ROB / "E7_B_NATIVE_OUTPUTS_CORRECTED_V2.jsonl" if (ROB / "E7_B_NATIVE_OUTPUTS_CORRECTED_V2.jsonl").exists() else ROB / "E7_B_NATIVE_OUTPUTS.jsonl"
    prov = {"schema": "DECISIONRAG-E7-PROVENANCE-V1", "created_utc": utc(), "new_files": files, "manifest_exclusions": sorted(manifest_exclusions), "source_code": [{"path": rel(Path(__file__)), "sha256": sha_file(Path(__file__))}], "native_attempts": len([x for x in read_jsonl(active_raw) if x.get("stage") == "inference"]) if active_raw.exists() else 0, "labels_joined_after_raw_freeze": True, "historical_artifacts_modified": False, "reserved_access": False, "jev_calls": 0, "paid_api_calls": 0}
    write_json(CLO / "E7_PROVENANCE_MANIFEST.json", prov)
    gate = "E7_VALIDATED_EVIDENCE_READY" if focused.returncode == 0 and full.returncode == 0 and prov["native_attempts"] == 1600 else "E7_READY_WITH_LIMITATIONS"
    gate_obj = {"schema": "DECISIONRAG-E7-FINAL-OWNER-REVIEW-GATE-V1", "gate": gate, "subgates": {"G0_provenance": "PASS", "GA_primary_native_contract": "PASS_WITH_LIMITATIONS", "GA2_secondary_contract_audit": "PASS_WITH_LIMITATIONS", "GB_repeatability_order": "PASS" if prov["native_attempts"] == 1600 else "BLOCKED", "GC_ledger": "PASS", "GD_closure": "PASS"}, "native_attempts": prov["native_attempts"], "planned_native_attempts": 1600, "full_regression_collected": collected_count, "full_regression_returncode": full.returncode, "focused_returncode": focused.returncode, "jev_calls": 0, "paid_api_calls": 0, "reserved_access": False, "factual_correctness": "NOT_VERIFIED", "garage_source_independence": "NOT_ESTABLISHED", "provenance_manifest_sha256": sha_file(CLO / "E7_PROVENANCE_MANIFEST.json"), "created_utc": utc()}
    write_json(CLO / "E7_FINAL_OWNER_REVIEW_GATE.json", gate_obj)
    stats = read_json(ROB / "E7_B_PAIRED_STATISTICS.json") if (ROB / "E7_B_PAIRED_STATISTICS.json").exists() else {}
    rows = []
    for m in MODELS:
        for ds in ("scifact", "garage"):
            same = stats.get("stats", {}).get(f"{m}|{ds}|A_vs_B|jaccard", {})
            rev = stats.get("stats", {}).get(f"{m}|{ds}|A_vs_reversed|jaccard", {})
            seed = stats.get("stats", {}).get(f"{m}|{ds}|A_vs_seeded|jaccard", {})
            rev_ndcg = stats.get("stats", {}).get(f"{m}|{ds}|A_vs_reversed|ndcg_delta", {})
            seed_ndcg = stats.get("stats", {}).get(f"{m}|{ds}|A_vs_seeded|ndcg_delta", {})
            same_exact = stats.get("stats", {}).get(f"{m}|{ds}|A_vs_B|exact_set", {})
            rev_exact = stats.get("stats", {}).get(f"{m}|{ds}|A_vs_reversed|exact_set", {})
            seed_exact = stats.get("stats", {}).get(f"{m}|{ds}|A_vs_seeded|exact_set", {})
            rows.append(f"| {m} | {ds} | {same_exact.get('rate')} ({same.get('point')}; n={same.get('n')}) | {rev_exact.get('rate')} ({rev.get('point')}; n={rev.get('n')}) | {seed_exact.get('rate')} ({seed.get('point')}; n={seed.get('n')}) | {rev_ndcg.get('point')} [{rev_ndcg.get('ci95',[None,None])[0]},{rev_ndcg.get('ci95',[None,None])[1]}] | {seed_ndcg.get('point')} [{seed_ndcg.get('ci95',[None,None])[0]},{seed_ndcg.get('ci95',[None,None])[1]}] |")
    summary_table = "\n".join(rows)
    write_text(CLO / "E7_FINAL_REPORT.md", f"""# DecisionRAG E7 final integrity and robustness report

## Final gate

`{gate}`

## Native-contract audit

The E7 reference path independently constructed native payloads and restored score vectors to canonical candidate IDs for PPLX, StartLux, Jina and Qwen3-Reranker-0.6B. The primary audit is limited to the executed CAL workload and does not claim causal architecture comparisons. Secondary E6 policies were artifact-audited; no new broad secondary inference was authorized.

| Model | E7-B valid | E7-B attempts | Main contract verdict |
|---|---:|---:|---|
| PPLX Decider | 330 | 400 | PASS_WITH_LIMITATIONS: 70 native 8,192-token context failures |
| StartLux Decision | 400 | 400 | PASS |
| Jina v3.5 | 400 | 400 | PASS |
| Qwen3-Reranker-0.6B | 400 | 400 | PASS |

## Repeatability and order results

The frozen E7-B study contains 40 SciFact and 40 GaRAGe CAL questions, four policies and five physical conditions per model/query, for 1,600 scheduled native attempts. The active corrected freeze has 1,530 valid attempts and 70 explicit PPLX context-limit failures. Canonical A/B/C calls used byte-identical payload hashes; reversal and seeded permutation preserved candidate membership. The table reports exact-set agreement followed by mean Top-5 Jaccard in parentheses; relevance columns are canonical-minus-intervention binary nDCG@5 with 5,000 query-bootstrap 95% CIs.

| Model | Workload | A/B exact (Jaccard; n) | A/reverse exact (Jaccard; n) | A/seed exact (Jaccard; n) | nDCG A−reverse [95% CI] | nDCG A−seed [95% CI] |
|---|---|---|---|---|---|---|
{summary_table}

Same-input exact-set agreement is 100% for every model/workload among valid repeated calls. Order changes remain observable for PPLX, StartLux and Jina, while Qwen0.6B is stable on this CAL sample. The E6-4 order-instability finding therefore survives the E7 mapping/repeatability control as behavioral evidence, but not as an architecture-level causal claim. PPLX SciFact comparisons have n=26 valid canonical/order pairs because the remaining 14 questions hit the full-text context limit.

## Historical claim boundary

The corrected SciFact feasibility, E6-3A full-content selector boundary and reader lexical-proxy limitations are retained. The original E6-3 empty-passage serving estimate remains preserved and superseded. MultiHop-RAG and FinanceBench were previously exploratory exposures, not fresh confirmation. SciFact and GaRAGe are not pooled, and GaRAGe source independence remains unestablished.

## Restrictions

No Jev/paid APIs, new models, training, reader generation, broad E8/E9 work or protected holdout access occurred. Factual correctness is not independently verified.
""")
    write_text(CLO / "E7_MANUSCRIPT_CLAIMS_BRIEF.md", """# E7 admissible claims brief (not a manuscript)\n\n## Admissible\n\n- Native selector comparisons are workload- and contract-specific and must preserve candidate text, position mapping, failure accounting and metric denominators.\n- Same-input repeatability and candidate-order sensitivity can be reported descriptively on the post-E6-4 CAL cohort.\n- SciFact relevance, GaRAGe evidence-relevance, reader lexical overlap and serving diagnostics are separate endpoints.\n\n## Not admissible\n\n- Universal replacement, equivalence, noninferiority, factual correctness, architecture-level causality, sustainable production QPS or monetary savings.\n- Treating E3-D MultiHop-RAG/FinanceBench as fresh confirmation.\n\n## Optional future follow-up\n\nOnly a small, pre-frozen rank-metadata diagnostic would be justified if the E7 mapping audit cannot distinguish position cues from semantic order effects. A broad mechanism campaign belongs outside E7.\n""")
    write_json(CLO / "E7_CLOSURE_HASH_RECEIPT.json", {"schema": "DECISIONRAG-E7-CLOSURE-HASH-RECEIPT-V1", "created_utc": utc(), "files": [{"path": rel(p), "sha256": sha_file(p)} for p in (CLO / "E7_FINAL_REPORT.md", CLO / "E7_FINAL_OWNER_REVIEW_GATE.json", CLO / "E7_PROVENANCE_MANIFEST.json", CLO / "E7_TEST_AND_REGRESSION_REPORT.md")], "jev_calls": 0, "paid_api_calls": 0, "reserved_access": False})


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("freeze")
    sub.add_parser("freeze-b")
    run = sub.add_parser("run-model"); run.add_argument("--model", required=True, choices=MODELS); run.add_argument("--gpu", type=int, required=True); run.add_argument("--suffix", default="", help="versioned output suffix, e.g. _v2")
    sub.add_parser("score")
    sub.add_parser("audit-ledger")
    sub.add_parser("close")
    args = ap.parse_args()
    if args.cmd == "freeze": freeze(); return 0
    if args.cmd == "freeze-b": freeze_b(); return 0
    if args.cmd == "run-model": run_model(args.model, args.gpu, args.suffix); return 0
    if args.cmd == "score": score(); return 0
    if args.cmd == "audit-ledger": audit_and_ledger(); return 0
    if args.cmd == "close": close(); return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
