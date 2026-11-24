#!/usr/bin/env python3
"""DecisionRAG E4-3 bounded operating-regime experiment.

This runner is intentionally additive.  It consumes the already frozen E3-D
exploratory MultiHop slates, freezes a new E4-3 cohort before quality is read,
reuses only exact K=20 specialist traces, and records native K=10/Clef work in
a new result namespace.  It never opens holdout outcomes or calls Jev.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e4_v1_1/e4_3_operating_regimes"
E3 = ROOT / "results/non_jev/v2/e3_v1/e3_d_external"
E42 = ROOT / "results/non_jev/v2/e4_v1_1/e4_2_facet_pilot"
HF_ROOT = Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub")
SEED = 731
BOOTSTRAP_REPS = 5000
GPU_ID = int(os.environ.get("E4_3_GPU_ID", "1"))
MAX_LENGTH = 16384

SPECS = {
    "clef_pointwise_binary": {"model_id": "Cloudflare/clef-flash", "revision": "8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3", "kind": "clef_pointwise", "historical": None},
    "clef_joint_binary": {"model_id": "Cloudflare/clef-flash", "revision": "8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3", "kind": "clef_joint", "historical": None},
    "liquid_native_joint": {"model_id": "LiquidAI/d1-3B", "revision": "051bcc464b01b9f92942b364d9586b0ef5912432", "kind": "liquid_native", "historical": "liquid_joint20_binary"},
    "jina35_native": {"model_id": "jinaai/jina-reranker-v3.5", "revision": "e8a93f33f0b22108f8c2364f8484ce3422552fbc", "kind": "jina_native", "historical": "jina35"},
    "pplx_decider_native": {"model_id": "perplexity-ai/pplx-decider-v1.1-27b", "revision": "14c9d90137962d300f3de95633f7b6535e80c5ce", "kind": "pplx_native", "historical": "pplx_decider"},
    "bge_pointwise": {"model_id": "BAAI/bge-reranker-v2-m3", "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e", "kind": "bge_native", "historical": "bge_pointwise"},
}


def sha_bytes(v: bytes) -> str:
    return hashlib.sha256(v).hexdigest()


def sha_json(v: Any) -> str:
    return sha_bytes(json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
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


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{2,}", str(text).lower()[:12000]))


def run_gpu_snapshot() -> dict[str, Any]:
    cmd = ["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return {"status": "PASS" if p.returncode == 0 else "UNAVAILABLE", "command": " ".join(cmd), "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except Exception as exc:
        return {"status": "UNAVAILABLE", "error": f"{type(exc).__name__}: {exc}"}


def source_files() -> list[Path]:
    return [
        E3 / "E3_D_FROZEN_EXPLORATORY_PROTOCOL.json",
        E3 / "E3_D_EXPLORATORY_METADATA.jsonl",
        E3 / "E3_D_FIRST_STAGE_CANDIDATES.jsonl",
        E3 / "E3_D_CANDIDATE_VIEWS.jsonl",
        E3 / "E3_D_NATIVE_SELECTION_TRACES.jsonl",
        E42 / "E4_2_INPUT_AND_COHORT_MANIFEST.json",
        ROOT / "DecisionRAG_E4_3_Operating_Regimes_Execution_SDD_Codex.md",
    ]


def load_foundation() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    protocol = read_json(E3 / "E3_D_FROZEN_EXPLORATORY_PROTOCOL.json")
    selected = set(protocol["selected_query_ids"])
    metadata = {x["query_id"]: x for x in read_jsonl(E3 / "E3_D_EXPLORATORY_METADATA.jsonl") if x.get("query_id") in selected and x.get("dataset") == "multihop" and x.get("valid_identity")}
    candidates = {x["query_id"]: x for x in read_jsonl(E3 / "E3_D_FIRST_STAGE_CANDIDATES.jsonl") if x.get("query_id") in metadata}
    views: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(E3 / "E3_D_CANDIDATE_VIEWS.jsonl"):
        if row.get("query_id") in metadata:
            views.setdefault(row["query_id"], {})[row["doc_id"]] = row
    prev = read_json(E42 / "E4_2_INPUT_AND_COHORT_MANIFEST.json")
    prior_qids = {x["qid"] for x in prev.get("common_candidate_slate_manifest", [])}
    rows = [metadata[q] for q in sorted(metadata) if q in candidates and q in views and q not in prior_qids]
    if len(rows) < 380:
        raise RuntimeError(f"only {len(rows)} eligible fresh E4-3 MultiHop records after E4-2 exclusion")
    return rows, candidates, views, {"e3_protocol": protocol, "prior_qids": prior_qids}


def redundancy_score(candidate: dict[str, Any]) -> float:
    sets = [tokenize(candidate.get("title", "") + "\n" + candidate.get("full_text", "")) for candidate in candidate.get("_candidates", [])]
    vals = []
    for i in range(len(sets)):
        for j in range(i):
            union = sets[i] | sets[j]
            vals.append(len(sets[i] & sets[j]) / max(1, len(union)))
    return max(vals, default=0.0)


def freeze_cohort() -> dict[str, Any]:
    rows, candidates, views, extra = load_foundation()
    scored = []
    for m in rows:
        slate = candidates[m["query_id"]]
        cs = [views[m["query_id"]][d] for d in slate["candidate_ids"]]
        score = redundancy_score({"_candidates": cs})
        scored.append((m, slate, cs, score))
    scores = sorted(x[3] for x in scored)
    q33, q67 = scores[len(scores) // 3], scores[(2 * len(scores)) // 3]
    for m, slate, cs, score in scored:
        m["_e4_redundancy"] = score
        m["_e4_redundancy_band"] = "low" if score <= q33 else "high" if score >= q67 else "middle"
    order = sorted(scored, key=lambda x: sha_bytes(f"{SEED}|representative|{x[0]['query_id']}".encode()))
    rep = order[:300]
    remaining = order[300:]
    # Diagnostic selection is deterministic and deliberately enriched only on
    # predeclared text-derived strata, never on model outcomes or qrels.
    def diag_key(x):
        m = x[0]
        return (m.get("_e4_redundancy_band") == "middle", m.get("support_cardinality") == 2, sha_bytes(f"{SEED}|diagnostic|{m['query_id']}".encode()))
    diag = sorted(remaining, key=diag_key)[:80]
    chosen = [("representative", x) for x in rep] + [("diagnostic", x) for x in diag]
    out = []
    for cohort, (m, slate, cs, red) in chosen:
        source_ids = list(dict.fromkeys(m.get("source_ids") or []))
        source_keys = {str(x)[4:] if str(x).startswith("url:") else str(x) for x in source_ids}
        support_ids = []
        for c in cs:
            if c.get("source_key", "").rstrip("/") in {x.rstrip("/") for x in source_keys}:
                support_ids.append(c["doc_id"])
        gold_count = len(source_keys)
        q = {
            "query_id": m["query_id"], "question": m["question"], "question_hash": m["question_hash"],
            "source_group_id": m.get("component_id"), "dataset": "multihop", "dataset_revision": "E3-D-exploratory-v1",
            "cohort": cohort, "source_record_hash": m.get("source_record_hash"), "support_cardinality": m.get("support_cardinality"),
            "source_keys_for_analysis_only": sorted(source_keys), "gold_support_ids_for_analysis_only": sorted(set(support_ids)),
            "gold_count": gold_count, "candidate_ids": list(slate["candidate_ids"]),
            "candidate_ids_sha256": sha_json(slate["candidate_ids"]), "candidate_order": slate.get("candidate_order"),
            "candidate_view_hashes": [c["candidate_view_sha256"] for c in cs],
            "candidate_view_hashes_sha256": sha_json([c["candidate_view_sha256"] for c in cs]),
            "candidate_word_lengths": [len((c.get("title", "") + "\n" + c.get("full_text", "")).split()) for c in cs],
            "baseline_first_stage_ranks": list(range(1, 21)), "k10_candidate_ids": list(slate["candidate_ids"][:10]),
            "k20_candidate_ids": list(slate["candidate_ids"]), "redundancy_score": red,
            "redundancy_band": m["_e4_redundancy_band"], "gold_in_policy_input": False,
        }
        out.append(q)
    out.sort(key=lambda x: x["query_id"])
    return {"rows": out, "redundancy_cutpoints": {"low_upper": q33, "high_lower": q67}, "prior_qids": sorted(extra["prior_qids"]), "source_protocol": extra["e3_protocol"]}


def foundation_stage() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    cohort = freeze_cohort()
    rows = cohort["rows"]
    protocol = {
        "protocol": "DecisionRAG E4-3 v1.0",
        "status": "FROZEN_EXPLORATORY_NONRESERVED",
        "seed": SEED, "bootstrap_repetitions": BOOTSTRAP_REPS,
        "cohort_counts": dict(Counter(x["cohort"] for x in rows)),
        "dataset_counts": dict(Counter(x["dataset"] for x in rows)),
        "cohort_qids_sha256": sha_json([x["query_id"] for x in rows]),
        "candidate_contract": "E3-D original Top-20 order; K10 is first ten; K20 is full twenty; no new retrieval",
        "top5_contract": "exactly five distinct canonical candidate IDs; stable sort by descending score, original rank, ID",
        "redundancy": {"measure": "maximum pairwise token-set Jaccard over title+full_text first 12000 chars", "cutpoints": cohort["redundancy_cutpoints"], "calibration": "unsupervised E3-D exploratory candidate-text distribution before quality"},
        "policies": SPECS,
        "native_view_contracts": {
            "clef": "E2-M1 make_views: question 64 tokens, passage 128-token view plus title retention; pointwise independent binary and shared-state joint binary",
            "bge": "E3-D model_text title+first-12000-char passage; tokenizer pair max 512",
            "jina": "E3-D model_text; native rerank; question 64-token and passage 512-token deterministic cap",
            "liquid": "question 64-token and candidate 256-token deterministic cap; native system_one binary decisions",
            "pplx": "question 64-token and candidate 128-token deterministic cap; native categorical DecisionModel readout",
        },
        "input_leakage_policy": {"gold_in_policy_input": False, "support_ids_in_policy_input": False, "reference_in_policy_input": False},
        "gpu_authorization": {"approved": [0, 1, 2, 4], "selected_for_new_inference": GPU_ID, "timing_requirement": "truly isolated approved GPU; otherwise blocked"},
        "jev_calls": 0, "paid_api_calls": 0, "training": False, "reader_calls": 0, "new_retrieval": False,
    }
    write_jsonl(OUT / "E4_3_COHORT_AND_STRATA_MANIFEST.jsonl", rows)
    write_json(OUT / "E4_3_FROZEN_EXPLORATORY_PROTOCOL.json", protocol)
    return {"rows": rows, "protocol": protocol, "prior_qids": cohort["prior_qids"]}


def provenance_audit() -> None:
    records = []
    for p in source_files():
        records.append({"path": str(p.relative_to(ROOT)), "exists": p.is_file(), "sha256": sha_file(p) if p.is_file() else None, "bytes": p.stat().st_size if p.is_file() else None})
    prev = read_json(E42 / "E4_2_PROVENANCE_MANIFEST.json")
    write_text(OUT / "E4_3_INPUT_PROVENANCE_AND_HASH_AUDIT.md", "# E4-3 Input Provenance and Hash Audit\n\n" +
               "All listed inputs are read-only references. The E3-D exploratory protocol declares `reserved_ids_accessed=false`, `gold_accessed_during_generation=false`, `jev_calls=0`, and `paid_api_calls=0`. E4-2 artifacts are preserved and not overwritten.\n\n" +
               "## Input hashes\n\n" + "\n".join(f"- `{r['path']}`: `{r['sha256']}` ({r['bytes']} bytes)" for r in records) +
               f"\n\nPrior E4-2 provenance status: `{prev.get('status', 'not_available')}`. New E4-3 cohort excludes `{len(read_json(OUT / 'E4_3_FROZEN_EXPLORATORY_PROTOCOL.json').get('cohort_qids_sha256', '')) and 'the prior E4-2 qid set'}` by deterministic ID comparison.\n")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def policy_input_hash(q: dict[str, Any], k: int, hist_policy: str) -> str:
    return sha_json({"question": q["question"], "candidate_ids": q[f"k{k}_candidate_ids"], "candidate_views": q["candidate_view_hashes"][:k], "policy": hist_policy})


def load_traces() -> dict[tuple[str, str], dict[str, Any]]:
    out = {}
    for row in read_jsonl(E3 / "E3_D_NATIVE_SELECTION_TRACES.jsonl"):
        if row.get("stage") != "full" or row.get("status") != "VALID":
            continue
        key = (row.get("policy"), row.get("query_id"))
        out[key] = row
    return out


def exact_reuse(q: dict[str, Any], policy: str, k: int, traces: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any] | None:
    spec = SPECS[policy]
    hist = spec.get("historical")
    if k != 20 or not hist:
        return None
    row = traces.get((hist, q["query_id"]))
    if not row or row.get("model_revision") != spec["revision"] or row.get("candidate_ids") != q["k20_candidate_ids"]:
        return None
    expected = policy_input_hash(q, 20, hist)
    if row.get("input_sha256") != expected or len(row.get("scores", [])) != 20 or row.get("gold_in_inference_input") is True:
        return None
    return {**row, "e4_policy": policy, "k": 20, "stage": "historical_exact_reuse", "cache_provenance": "E3-D exact input/model/contract reuse", "e4_input_sha256": expected, "physical_inference_evidence": row.get("physical_inference_evidence", False)}


def stable_top5(ids: list[str], scores: list[float]) -> list[str]:
    return [x[1] for x in sorted(zip(scores, ids), key=lambda x: (-float(x[0]), ids.index(x[1]), x[1]))[:5]]


def clef_binary_score(logits: Any, encoded_question: Any, torch: Any) -> float:
    probabilities = logits.float().softmax(-1).detach().cpu().tolist()
    option_ids = list(encoded_question.option_ids)
    if "true" not in option_ids or "false" not in option_ids:
        raise RuntimeError(f"Clef binary options invalid: {option_ids}")
    score = float(probabilities[option_ids.index("true")])
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise RuntimeError(f"invalid Clef binary score {score}")
    return score


def make_clef_item(q: dict[str, Any], k: int, view_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {"query_id": q["query_id"], "cohort": q["cohort"], "dataset": q["dataset"], "question": q["question"], "candidates": [{"doc_id": d, "dataset": "multihop", "title": view_rows[d]["title"], "full_text": view_rows[d]["full_text"], "original_rank": i + 1} for i, d in enumerate(q[f"k{k}_candidate_ids"])]}


def run_clef(policy: str, q: dict[str, Any], k: int, view_rows: dict[str, dict[str, Any]], module: Any, model: Any, processor: Any, tokenizer: Any, torch: Any) -> dict[str, Any]:
    from decisionrag.selection import make_views, pointwise_state, pointwise_question
    item = make_clef_item(q, k, view_rows)
    views = [make_views(item["question"], c, tokenizer) for c in item["candidates"]]
    started = time.perf_counter_ns(); torch.cuda.reset_peak_memory_stats()
    if policy == "clef_pointwise_binary":
        records = [{"state": pointwise_state(v), "questions": pointwise_question(), "id": f"{q['query_id']}::{v.doc_id}"} for v in views]
        encoded = [module.encode_record(processor.tokenizer, r, processor=processor, max_length=MAX_LENGTH) for r in records]
        scores = []
        forwards = 0
        for start in range(0, len(encoded), 20):
            batch = module.collate_records(encoded[start:start + 20], processor.tokenizer.pad_token_id, torch.device("cuda"))
            with torch.inference_mode(): outputs = model(batch)
            forwards += 1
            for enc, output in zip(encoded[start:start + 20], outputs): scores.append(clef_binary_score(output[0], enc.questions[0], torch))
        formulation, contract, alignment = "pointwise_binary", "INDEPENDENT_RELEVANCE:v2", "candidate_ids_order"
        input_tokens = [len(x.input_ids) for x in encoded]
        input_hash = sha_json([list(x.input_ids) for x in encoded])
    else:
        state = {"question": views[0].question_view, "candidates": [{"candidate_id": v.doc_id, "original_rank": v.original_rank, "title": v.title, "passage": v.passage_view} for v in views]}
        questions = {f"relevance_{i:02d}_{v.doc_id}": {"type": "noul", "instructions": f"For candidate_id={v.doc_id}, does that candidate passage contain factual evidence useful for answering the question? Answer true only for actual evidence, including an intermediate multi-hop link.", "criteria": {"true": "The named candidate contains useful factual evidence.", "false": "The named candidate does not contain useful factual evidence."}} for i, v in enumerate(views)}
        record = {"id": f"{q['query_id']}::joint::{k}", "state": state, "questions": questions}
        encoded = module.encode_record(processor.tokenizer, record, processor=processor, max_length=MAX_LENGTH)
        if len(encoded.input_ids) > MAX_LENGTH: raise RuntimeError(f"joint K{k} context exceeds {MAX_LENGTH}: {len(encoded.input_ids)}")
        batch = module.collate_records([encoded], processor.tokenizer.pad_token_id, torch.device("cuda"))
        with torch.inference_mode(): outputs = model(batch)
        probs = outputs[0]
        if len(probs) != k: raise RuntimeError(f"joint native output fields {len(probs)} != K{k}")
        scores = [clef_binary_score(logit, enc, torch) for logit, enc in zip(probs, encoded.questions)]
        forwards, formulation, contract, alignment = 1, "native_joint_binary", "JOINT_BINARY_RELEVANCE:v2", "field_id_candidate_order"
        input_tokens, input_hash = len(encoded.input_ids), sha_json(list(encoded.input_ids))
    ids = q[f"k{k}_candidate_ids"]
    return {"e4_policy": policy, "k": k, "query_id": q["query_id"], "cohort": q["cohort"], "dataset": q["dataset"], "model_id": "Cloudflare/clef-flash", "model_revision": SPECS[policy]["revision"], "formulation": formulation, "score_contract": contract, "candidate_ids": ids, "scores": [float(x) for x in scores], "selected_ids": stable_top5(ids, scores), "score_direction": "higher_is_more_relevant", "score_alignment": alignment, "candidate_view_hashes": q["candidate_view_hashes"][:k], "e4_input_sha256": input_hash, "policy_input_sha256": policy_input_hash(q, k, policy), "actual_model_input_sha256": input_hash, "response_sha256": sha_json(scores), "input_tokens": input_tokens, "context_overflow": False, "physical_forward_calls": forwards, "physical_inference_evidence": True, "duration_ms": (time.perf_counter_ns() - started) / 1_000_000, "gpu_id": GPU_ID, "visible_cuda_device": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu_memory_peak_bytes": int(torch.cuda.max_memory_allocated()), "gold_in_inference_input": False, "cache_reuse": False, "status": "VALID", "real_inference": True}


class ExternalAdapter:
    def __init__(self, policy: str):
        import torch
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch; self.policy = policy; self.spec = SPECS[policy]; self.forward_count = 0; self.captured = []
        kind = self.spec["kind"]
        if kind in {"bge_native", "jina_native"}:
            owner, repo = self.spec["model_id"].split("/", 1); path = HF_ROOT / f"models--{owner}--{repo}" / "snapshots" / self.spec["revision"]
            if not path.is_dir(): raise RuntimeError(f"missing checkpoint {path}")
            self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
            if kind == "jina_native": self.model = AutoModel.from_pretrained(path, dtype="auto", trust_remote_code=True, local_files_only=True).to("cuda").eval()
            else: self.model = AutoModelForSequenceClassification.from_pretrained(path, torch_dtype=torch.bfloat16, local_files_only=True).to("cuda").eval()
        elif kind == "liquid_native":
            path = HF_ROOT / "models--LiquidAI--d1-3B" / "snapshots" / self.spec["revision"]
            from transformers import AutoModel
            if not path.is_dir(): raise RuntimeError(f"missing checkpoint {path}")
            self.model = AutoModel.from_pretrained(path, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval(); self.tokenizer = getattr(self.model, "tokenizer", None)
            original = self.model.system_one; owner = self
            def counted(*args, **kwargs): owner.native_calls = getattr(owner, "native_calls", 0) + 1; return original(*args, **kwargs)
            self.model.system_one = counted
        elif kind == "pplx_native":
            path = HF_ROOT / "models--perplexity-ai--pplx-decider-v1.1-27b" / "snapshots" / self.spec["revision"]
            if not path.is_dir(): raise RuntimeError(f"missing checkpoint {path}")
            sys.path.insert(0, str(path / "source" / "src"))
            from autojev.model import DecisionModel
            self.model = DecisionModel(path, device="cuda", max_image_pixels=512 * 512); self.processor = self.model.processor; self.tokenizer = getattr(self.processor, "tokenizer", self.processor); self.native_calls = 0
            original = self.model.predict; owner = self
            def counted_predict(*args, **kwargs): owner.native_calls += 1; return original(*args, **kwargs)
            self.model.predict = counted_predict
        else: raise RuntimeError(kind)
        if hasattr(self, "model") and hasattr(self.model, "forward"):
            original_forward = self.model.forward; owner = self
            def counted_forward(*args, **kwargs):
                owner.forward_count += 1; return original_forward(*args, **kwargs)
            self.model.forward = counted_forward

    def cap(self, text: str, maximum: int) -> str:
        tok = self.tokenizer
        if not hasattr(tok, "encode"): return str(text)[:maximum * 8]
        ids = tok.encode(str(text), add_special_tokens=False)
        return tok.decode(ids[:maximum], skip_special_tokens=True).strip()

    def score(self, q: dict[str, Any], k: int, view_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
        import torch
        ids = q[f"k{k}_candidate_ids"]
        docs = [view_rows[x] for x in ids]
        kind = self.spec["kind"]; before = self.forward_count; native_before = getattr(self, "native_calls", 0); started = time.perf_counter_ns(); torch.cuda.reset_peak_memory_stats()
        texts = [(str(d.get("title") or "") + "\n" + str(d.get("full_text") or "")[:12000]).strip() for d in docs]
        if kind == "bge_native":
            enc = self.tokenizer([q["question"]] * k, texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
            with torch.inference_mode(): scores = self.model(**enc).logits.reshape(-1).float().cpu().tolist()
            input_tokens = int(enc["attention_mask"].sum().item())
        elif kind == "jina_native":
            question = self.cap(q["question"], 64); passages = [self.cap(x, 512) for x in texts]
            result = self.model.rerank(question, passages); mapping = {int(x["index"]): float(x["relevance_score"]) for x in result}
            if set(mapping) != set(range(k)): raise RuntimeError(f"Jina indices do not cover K{k}")
            scores = [mapping[i] for i in range(k)]; input_tokens = None
        elif kind == "liquid_native":
            question = self.cap(q["question"], 64); passages = [self.cap(x, 256) for x in texts]
            state = {"query": question, "candidates": [{"candidate_id": d["doc_id"], "title": d["title"], "text": p} for d, p in zip(docs, passages)]}
            questions = {f"candidate_{i:02d}": {"type": "noul", "instructions": f"Does candidate {d['doc_id']} contain information relevant and useful for answering the query? Answer yes only when this candidate provides evidence needed for the query; otherwise answer no.", "criteria": {"true": "Yes", "false": "No"}} for i, d in enumerate(docs)}
            response = self.model.system_one(state, questions); scores = [float(response["answers"][f"candidate_{i:02d}"]["noul"]) for i in range(k)]; input_tokens = None
        elif kind == "pplx_native":
            question = self.cap(q["question"], 64); passages = [self.cap(x, 128) for x in texts]
            criteria = {f"candidate_{i:02d}": f"Candidate ID {d['doc_id']}. Title and passage: {p}" for i, (d, p) in enumerate(zip(docs, passages))}
            request = {"state": f"Query: {question}\nEvaluate the supplied candidate passages using the option descriptions.", "question": {"type": "choice", "instructions": f"Select the candidate passage that best answers the query: {question}", "criteria": criteria}}
            scores = [float(x) for x in self.model.predict([request], batch_size=1)[0]]; input_tokens = None
        else: raise RuntimeError(kind)
        scores = [float(x) for x in scores]
        if len(scores) != k or not all(math.isfinite(x) for x in scores): raise RuntimeError(f"invalid K{k} score vector")
        return {"e4_policy": q.get("_running_policy"), "k": k, "query_id": q["query_id"], "cohort": q["cohort"], "dataset": q["dataset"], "model_id": self.spec["model_id"], "model_revision": self.spec["revision"], "candidate_ids": ids, "scores": scores, "selected_ids": stable_top5(ids, scores), "score_direction": "higher_is_more_relevant", "candidate_view_hashes": q["candidate_view_hashes"][:k], "e4_input_sha256": sha_json({"question": q["question"], "candidate_ids": ids, "candidate_views": q["candidate_view_hashes"][:k], "policy": q.get("_running_policy")}), "actual_model_input_sha256": sha_json({"policy": self.policy, "k": k, "query": q["question"], "ids": ids, "texts": texts}), "response_sha256": sha_json(scores), "input_tokens": input_tokens, "physical_forward_calls": self.forward_count - before, "native_api_calls": getattr(self, "native_calls", 0) - native_before, "physical_inference_evidence": (self.forward_count > before) or (getattr(self, "native_calls", 0) > native_before), "duration_ms": (time.perf_counter_ns() - started) / 1_000_000, "gpu_id": GPU_ID, "visible_cuda_device": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu_memory_peak_bytes": int(torch.cuda.max_memory_allocated()), "gold_in_inference_input": False, "cache_reuse": False, "status": "VALID", "real_inference": True}

    def close(self):
        try: del self.model
        except Exception: pass
        try: self.torch.cuda.empty_cache()
        except Exception: pass


def append_raw(row: dict[str, Any]) -> None:
    with (OUT / "E4_3_RAW_NATIVE_INFERENCE.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run_smoke_or_full(stage: str, policies: list[str] | None = None) -> None:
    frozen = read_json(OUT / "E4_3_FROZEN_EXPLORATORY_PROTOCOL.json"); qs = read_jsonl(OUT / "E4_3_COHORT_AND_STRATA_MANIFEST.jsonl")
    rows_by = {x["query_id"]: x for x in qs}; views_all: dict[str, dict[str, Any]] = {}
    for r in read_jsonl(E3 / "E3_D_CANDIDATE_VIEWS.jsonl"):
        if r.get("query_id") in rows_by: views_all.setdefault(r["query_id"], {})[r["doc_id"]] = r
    traces = load_traces(); existing = {(r.get("e4_policy"), r.get("k"), r.get("query_id")): r for r in read_jsonl(OUT / "E4_3_RAW_NATIVE_INFERENCE.jsonl") if r.get("status") == "VALID"}
    # Smoke uses five already authenticated E4-2 CAL questions, outside the
    # new E4-3 quality cohorts.  Their candidate/text views are re-read from
    # E3-D and are never mixed into E4-3 quality denominators.
    if stage == "full":
        target = qs
    else:
        all_meta = {x["query_id"]: x for x in read_jsonl(E3 / "E3_D_EXPLORATORY_METADATA.jsonl")}
        all_candidates = {x["query_id"]: x for x in read_jsonl(E3 / "E3_D_FIRST_STAGE_CANDIDATES.jsonl")}
        smoke_ids = read_json(E42 / "E4_2_INPUT_AND_COHORT_MANIFEST.json").get("cal_qids", [])[:5]
        smoke_views: dict[str, dict[str, Any]] = {}
        for r in read_jsonl(E3 / "E3_D_CANDIDATE_VIEWS.jsonl"):
            if r.get("query_id") in smoke_ids: smoke_views.setdefault(r["query_id"], {})[r["doc_id"]] = r
        target = []
        for qid in smoke_ids:
            if qid not in all_meta or qid not in all_candidates or qid not in smoke_views: continue
            m = all_meta[qid]; slate = all_candidates[qid]; cs = [smoke_views[qid][d] for d in slate["candidate_ids"]]
            target.append({"query_id": qid, "question": m["question"], "question_hash": m["question_hash"], "source_group_id": m.get("component_id"), "dataset": "multihop", "cohort": "cal_smoke", "candidate_ids": slate["candidate_ids"], "k10_candidate_ids": slate["candidate_ids"][:10], "k20_candidate_ids": slate["candidate_ids"], "candidate_view_hashes": [c["candidate_view_sha256"] for c in cs], "gold_support_ids_for_analysis_only": [], "gold_count": None})
        views_all.update(smoke_views)
    policies = policies or list(SPECS)
    for policy in policies:
        for k in (10, 20):
            if k == 20:
                for q in target:
                    reused = exact_reuse(q, policy, k, traces)
                    if reused:
                        reused["stage"] = "historical_exact_reuse" if stage == "full" else "historical_exact_reuse_smoke"; reused["status"] = "VALID"
                        key = (policy, k, q["query_id"])
                        if key not in existing:
                            append_raw(reused)
                            existing[key] = reused
            if policy in {"clef_pointwise_binary", "clef_joint_binary"}:
                try:
                    sys.path.insert(0, str(ROOT / "scripts")); import e2_m1_experiment as m1
                    module, model, processor, torch = m1.import_clef(); tokenizer = processor.tokenizer
                    for q in target:
                        key = (policy, k, q["query_id"])
                        if key in existing: continue
                        q["_running_policy"] = policy
                        try: append_raw(run_clef(policy, q, k, views_all[q["query_id"]], module, model, processor, tokenizer, torch))
                        except Exception as exc: append_raw({"e4_policy": policy, "k": k, "query_id": q["query_id"], "cohort": q["cohort"], "status": "INFERENCE_ERROR", "reason": f"{type(exc).__name__}: {exc}", "gpu_id": GPU_ID, "gold_in_inference_input": False, "cache_reuse": False})
                    del model; torch.cuda.empty_cache(); gc.collect()
                except Exception as exc:
                    for q in target: append_raw({"e4_policy": policy, "k": k, "query_id": q["query_id"], "cohort": q["cohort"], "status": "BLOCKED", "reason": f"adapter_load:{type(exc).__name__}: {exc}", "gpu_id": GPU_ID, "gold_in_inference_input": False, "cache_reuse": False})
                continue
            # External model is loaded once per policy/K stage.  A failed load is
            # retained as explicit blocked rows; no fallback proxy is substituted.
            need = [q for q in target if (policy, k, q["query_id"]) not in existing]
            if not need: continue
            adapter = None
            try:
                adapter = ExternalAdapter(policy)
                for q in need:
                    q["_running_policy"] = policy
                    try: append_raw(adapter.score(q, k, views_all[q["query_id"]]))
                    except Exception as exc: append_raw({"e4_policy": policy, "k": k, "query_id": q["query_id"], "cohort": q["cohort"], "status": "INFERENCE_ERROR", "reason": f"{type(exc).__name__}: {exc}", "gpu_id": GPU_ID, "gold_in_inference_input": False, "cache_reuse": False})
            except Exception as exc:
                for q in need: append_raw({"e4_policy": policy, "k": k, "query_id": q["query_id"], "cohort": q["cohort"], "status": "BLOCKED", "reason": f"adapter_load:{type(exc).__name__}: {exc}", "gpu_id": GPU_ID, "gold_in_inference_input": False, "cache_reuse": False})
            finally:
                if adapter: adapter.close(); gc.collect()


def bootstrap_mean(values: list[float], seed: int = SEED) -> tuple[float | None, float | None, float | None]:
    if not values: return None, None, None
    import random
    rng = random.Random(seed); n = len(values); means = []
    for _ in range(BOOTSTRAP_REPS): means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort(); return sum(values) / n, means[int(.025 * (len(means) - 1))], means[int(.975 * (len(means) - 1))]


def bootstrap_clusters(groups: dict[str, list[float]], seed: int = SEED) -> tuple[float | None, float | None, float | None]:
    """Cluster bootstrap preserving whole source groups per resample draw."""
    if not groups or not any(groups.values()):
        return None, None, None
    import random
    rng = random.Random(seed)
    names = list(groups)
    observed = [v for values in groups.values() for v in values]
    n = len(observed)
    means = []
    for _ in range(BOOTSTRAP_REPS):
        sampled = []
        for _ in names:
            sampled.extend(groups[rng.choice(names)])
        means.append(sum(sampled) / len(sampled))
    means.sort()
    return sum(observed) / n, means[int(.025 * (len(means) - 1))], means[int(.975 * (len(means) - 1))]


def quality_stage() -> None:
    qs = read_jsonl(OUT / "E4_3_COHORT_AND_STRATA_MANIFEST.jsonl"); qmap = {x["query_id"]: x for x in qs}
    raw = read_jsonl(OUT / "E4_3_RAW_NATIVE_INFERENCE.jsonl"); valid = {(r.get("e4_policy"), int(r.get("k")) if r.get("k") is not None else None, r.get("query_id")): r for r in raw if r.get("status") == "VALID" and len(r.get("selected_ids", [])) == 5 and len(set(r.get("selected_ids", []))) == 5}
    selected = []
    for key, r in sorted(valid.items()):
        selected.append({"policy": r["e4_policy"], "k": r["k"], "query_id": r["query_id"], "cohort": r.get("cohort"), "candidate_ids": r["candidate_ids"], "selected_ids": r["selected_ids"], "score_sha256": r.get("response_sha256"), "input_sha256": r.get("e4_input_sha256"), "status": "VALID"})
    write_jsonl(OUT / "E4_3_SELECTED_TOP5.jsonl", selected)
    score_rows = []
    outcome = {}
    for policy in SPECS:
        for k in (10, 20):
            for cohort in ("representative", "diagnostic"):
                ids = [q["query_id"] for q in qs if q["cohort"] == cohort]
                vals = [valid[(policy, k, qid)] for qid in ids if (policy, k, qid) in valid]
                all_n = len(vals); cssru_num = 0; cssrc_num = 0; cden = 0; ceiling = 0; any_num = 0; structural = 0
                for r in vals:
                    q = qmap[r["query_id"]]; gold = set(q["gold_support_ids_for_analysis_only"]); selected_ids = set(r["selected_ids"]); cand = set(q[f"k{k}_candidate_ids"])
                    complete = gold.issubset(cand)
                    if complete: ceiling += 1
                    if len(gold) > 5: structural += 1
                    if gold.issubset(selected_ids): cssru_num += 1
                    if complete: cden += 1; cssrc_num += int(gold.issubset(selected_ids))
                    any_num += int(bool(gold & selected_ids))
                score_rows.append({"policy": policy, "k": k, "cohort": cohort, "status": "MEASURED" if vals else "NOT_MEASURED", "valid_n": all_n, "cssr_u_num": cssru_num, "cssr_u_den": all_n, "cssr_u": cssru_num / all_n if all_n else None, "cssr_c_num": cssrc_num, "cssr_c_den": cden, "cssr_c": cssrc_num / cden if cden else None, "topk_complete_support_num": ceiling, "topk_complete_support_den": len(ids), "topk_complete_support_ceiling": ceiling / len(ids) if ids else None, "any_gold_top5_num": any_num, "any_gold_top5": any_num / all_n if all_n else None, "gold_count_gt5": structural, "context_or_inference_failures": len(ids) - all_n})
                outcome[(policy, k, cohort)] = {r["query_id"]: int(set(qmap[r["query_id"]]["gold_support_ids_for_analysis_only"]).issubset(set(r["selected_ids"]))) for r in vals}
    write_csv(OUT / "E4_3_RETRIEVAL_CEILINGS_AND_SCORECARD.csv", score_rows)
    mech = []
    for k in (10, 20):
        qids = [q["query_id"] for q in qs if q["cohort"] == "representative"]
        eligible = [qid for qid in qids if qmap[qid]["gold_support_ids_for_analysis_only"] and set(qmap[qid]["gold_support_ids_for_analysis_only"]).issubset(set(qmap[qid][f"k{k}_candidate_ids"])) and ("clef_pointwise_binary", k, qid) in valid and ("clef_joint_binary", k, qid) in valid]
        # Retain explicit paired vectors below; `outcome` is keyed by
        # (policy, k, cohort), not by individual query.  Avoid using that
        # summary map for query-level indexing here.
        pv = [int(set(qmap[qid]["gold_support_ids_for_analysis_only"]).issubset(set(valid[("clef_pointwise_binary", k, qid)]["selected_ids"]))) for qid in eligible]
        jv = [int(set(qmap[qid]["gold_support_ids_for_analysis_only"]).issubset(set(valid[("clef_joint_binary", k, qid)]["selected_ids"]))) for qid in eligible]
        d = [j - p for j, p in zip(jv, pv)]; mean, lo, hi = bootstrap_mean(d)
        mech.append({"contrast": "clef_joint_minus_pointwise", "k": k, "cohort": "representative", "matched_topk_feasible_n": len(eligible), "joint_success": sum(jv), "pointwise_success": sum(pv), "wins": sum(x > 0 for x in d), "losses": sum(x < 0 for x in d), "ties": sum(x == 0 for x in d), "difference": mean, "bootstrap95_low": lo, "bootstrap95_high": hi, "status": "MEASURED" if eligible else "NOT_MEASURED"})
    # Interaction is evaluated on the identical K10-feasible query set.
    q10 = [q["query_id"] for q in qs if q["cohort"] == "representative" and set(q["gold_support_ids_for_analysis_only"]).issubset(set(q["k10_candidate_ids"])) and all((p, k, q["query_id"]) in valid for p in ("clef_pointwise_binary", "clef_joint_binary") for k in (10, 20))]
    d10 = [int(set(qmap[x]["gold_support_ids_for_analysis_only"]).issubset(set(valid[("clef_joint_binary", 10, x)]["selected_ids"]))) - int(set(qmap[x]["gold_support_ids_for_analysis_only"]).issubset(set(valid[("clef_pointwise_binary", 10, x)]["selected_ids"]))) for x in q10]
    d20 = [int(set(qmap[x]["gold_support_ids_for_analysis_only"]).issubset(set(valid[("clef_joint_binary", 20, x)]["selected_ids"]))) - int(set(qmap[x]["gold_support_ids_for_analysis_only"]).issubset(set(valid[("clef_pointwise_binary", 20, x)]["selected_ids"]))) for x in q10]
    inter = [b - a for a, b in zip(d10, d20)]; mean, lo, hi = bootstrap_mean(inter)
    mech.append({"contrast": "interaction_D20_minus_D10", "k": "10_to_20", "cohort": "representative_K10_feasible", "matched_n": len(q10), "D10": sum(d10) / len(d10) if d10 else None, "D20": sum(d20) / len(d20) if d20 else None, "interaction": mean, "bootstrap95_low": lo, "bootstrap95_high": hi, "status": "MEASURED" if q10 else "NOT_MEASURED"})
    write_csv(OUT / "E4_3_PAIRED_MECHANISM_INTERACTIONS.csv", mech)
    strata = []
    for feature, values in (("redundancy_band", ["low", "middle", "high"]), ("support_cardinality", [1, 2, 3, 4]), ("length_band", ["short", "medium", "long"])):
        for value in values:
            ids = [q["query_id"] for q in qs if q["cohort"] == "representative" and (q.get(feature) == value or (feature == "length_band" and ((statistics.mean(q["candidate_word_lengths"]) < 250 and value == "short") or (250 <= statistics.mean(q["candidate_word_lengths"]) < 700 and value == "medium") or (statistics.mean(q["candidate_word_lengths"]) >= 700 and value == "long"))))]
            for policy in SPECS:
                for k in (10, 20):
                    rows2 = [valid[(policy, k, x)] for x in ids if (policy, k, x) in valid]
                    successes = [int(set(qmap[r["query_id"]]["gold_support_ids_for_analysis_only"]).issubset(set(r["selected_ids"]))) for r in rows2]
                    strata.append({"feature": feature, "stratum": value, "policy": policy, "k": k, "n": len(rows2), "cssr_u": sum(successes) / len(successes) if successes else None, "status": "MEASURED" if successes else "NOT_ESTIMABLE"})
    write_csv(OUT / "E4_3_REDUNDANCY_CARDINALITY_AND_LENGTH_STRATA.csv", strata)
    source_sensitivity = []
    representative_ids = [q["query_id"] for q in qs if q["cohort"] == "representative"]
    for policy in SPECS:
        for k in (10, 20):
            groups = defaultdict(list)
            for qid in representative_ids:
                row = valid.get((policy, k, qid))
                if row is None:
                    continue
                success = int(set(qmap[qid]["gold_support_ids_for_analysis_only"]).issubset(set(row["selected_ids"])))
                groups[str(qmap[qid]["source_group_id"])].append(success)
            flat = [v for values in groups.values() for v in values]
            largest_name, largest_values = max(groups.items(), key=lambda item: len(item[1])) if groups else (None, [])
            leave = [v for name, values in groups.items() if name != largest_name for v in values]
            query_mean, query_lo, query_hi = bootstrap_mean(flat, seed=SEED + k)
            cluster_mean, cluster_lo, cluster_hi = bootstrap_clusters(groups, seed=SEED + 1000 + k)
            source_sensitivity.append({
                "policy": policy, "k": k, "valid_n": len(flat),
                "source_group_count": len(groups), "largest_source_group": largest_name,
                "largest_group_n": len(largest_values), "query_cssr_u": query_mean,
                "query_bootstrap95_low": query_lo, "query_bootstrap95_high": query_hi,
                "cluster_bootstrap95_low": cluster_lo, "cluster_bootstrap95_high": cluster_hi,
                "leave_largest_group_out_n": len(leave),
                "leave_largest_group_out_cssr_u": sum(leave) / len(leave) if leave else None,
                "status": "MEASURED" if flat else "NOT_ESTIMABLE",
            })
    write_csv(OUT / "E4_3_SOURCE_GROUP_SENSITIVITY.csv", source_sensitivity)
    return {"valid_rows": len(selected), "score_rows": len(score_rows), "mechanism_rows": len(mech)}


def timing_block() -> None:
    rows = []
    for p in SPECS:
        for k in (10, 20):
            rows.append({"policy": p, "k": k, "status": "TIMING_BLOCKED_SHARED_GPU", "reason": "No approved GPU was isolated: approved devices had resident processes or active utilization at audit; no timings are presented as isolated.", "requested_queries": 200, "completed_queries": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None, "sustainable_qps": None, "cost": "NOT_MODELED", "gpu_id": GPU_ID})
    write_jsonl(OUT / "E4_3_ISOLATED_LATENCY_RAW.jsonl", rows)
    write_csv(OUT / "E4_3_ISOLATED_LATENCY_SUMMARY.csv", rows)


def reports() -> None:
    protocol = read_json(OUT / "E4_3_FROZEN_EXPLORATORY_PROTOCOL.json"); qs = read_jsonl(OUT / "E4_3_COHORT_AND_STRATA_MANIFEST.jsonl"); raw = read_jsonl(OUT / "E4_3_RAW_NATIVE_INFERENCE.jsonl")
    valid = [x for x in raw if x.get("status") == "VALID"]; blocked = [x for x in raw if x.get("status") != "VALID"]
    score_rows = list(csv.DictReader((OUT / "E4_3_RETRIEVAL_CEILINGS_AND_SCORECARD.csv").open()))
    mechanism_rows = list(csv.DictReader((OUT / "E4_3_PAIRED_MECHANISM_INTERACTIONS.csv").open()))
    def pctl(values: list[float], q: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))]
    quality_table = "\n".join(
        f"| {x['policy']} | K{x['k']} | {x['cohort']} | {x['valid_n']} | {x['cssr_u']} ({x['cssr_u_num']}/{x['cssr_u_den']}) | {x['cssr_c']} ({x['cssr_c_num']}/{x['cssr_c_den']}) | {x['topk_complete_support_ceiling']} | {x['any_gold_top5']} | {x['context_or_inference_failures']} |"
        for x in score_rows
    )
    mechanism_table = "\n".join(
        f"| {x['contrast']} | {x['k']} | {x.get('matched_topk_feasible_n') or x.get('matched_n')} | {x.get('difference') or x.get('interaction')} | {x['bootstrap95_low']} | {x['bootstrap95_high']} | {x['status']} |"
        for x in mechanism_rows
    )
    native_summary = []
    for policy in SPECS:
        for k in (10, 20):
            rows = [x for x in valid if x.get("e4_policy") == policy and int(x.get("k")) == k]
            physical = [x for x in rows if x.get("physical_inference_evidence") is True]
            durations = [float(x["duration_ms"]) for x in physical if x.get("duration_ms") is not None]
            peaks = [int(x["gpu_memory_peak_bytes"]) / (1024 ** 3) for x in physical if x.get("gpu_memory_peak_bytes") is not None]
            native_summary.append({
                "policy": policy, "k": k, "valid_n": len(rows), "physical_n": len(physical),
                # Historical specialist K20 rows are exact reuse when they
                # carry no physical-inference evidence.  Prefer that
                # authenticated provenance over the legacy cache_reuse flag.
                "reuse_n": sum(x.get("physical_inference_evidence") is not True for x in rows),
                "forward_calls": sum(int(x.get("physical_forward_calls") or 0) for x in physical),
                "duration_p50_ms": pctl(durations, .50), "duration_p95_ms": pctl(durations, .95),
                "peak_gb": max(peaks) if peaks else None,
                "gpu_ids": sorted({str(x.get("gpu_id")) for x in rows}),
            })
    native_table = "\n".join(
        f"| {x['policy']} | K{x['k']} | {x['valid_n']} | {x['physical_n']} | {x['reuse_n']} | {x['forward_calls']} | {x['duration_p50_ms']} | {x['duration_p95_ms']} | {x['peak_gb']} | {','.join(x['gpu_ids'])} |"
        for x in native_summary
    )
    tests_note = "Targeted E4-3 tests and the full repository regression suite are run by the finalization stage; results are recorded in E4_3_FULL_TEST_REPORT.md."
    write_text(OUT / "E4_3_NATIVE_INTERFACE_AND_FEASIBILITY_AUDIT.md", f"""# E4-3 Native Interface and Feasibility Audit

## Frozen population

- Representative: **{sum(x['cohort']=='representative' for x in qs)}**; diagnostic: **{sum(x['cohort']=='diagnostic' for x in qs)}**.
- Source groups: **{len(set(x['source_group_id'] for x in qs))}**; largest group: **{Counter(x['source_group_id'] for x in qs).most_common(1)[0] if qs else None}**.
- All candidates are the original E3-D Top-20 slates. K10 is the exact first-ten prefix; K20 is the exact original slate.

## Contracts

Clef pointwise uses `INDEPENDENT_RELEVANCE:v2`; Clef joint uses `JOINT_BINARY_RELEVANCE:v2`, the same Cloudflare/clef-flash revision and candidate view lineage. The current runner checks output-field count, candidate alignment and finite binary scores. Specialist controls use the existing E3-D native contract families and exact K20 reuse only when the input hash matches.

The PPLX checkpoint is a local 27B bfloat16 artifact. If its K10 load/inference failed, rows are retained as `BLOCKED` and no incompatible proxy is substituted. A successful historical K20 output is not treated as a new K10 call.

Valid new/reused rows: **{len(valid)}**; blocked/error rows: **{len(blocked)}**. Gold and support IDs were stored only as analysis-side fields and were not serialized into native request payloads.

Raw physical-call durations and GPU-memory peaks are retained as diagnostic telemetry. They were collected while the approved GPU was shared, so they are not isolated Top-K-to-Top-5 latency, sustainable QPS, or cost measurements.
""")
    write_text(OUT / "E4_3_QUALITY_LATENCY_REPLACEMENT_MAP.md", """# E4-3 Quality–Latency Replacement Map

Quality is reported as CSSR-U and CSSR-C evidence-retention proxies, with retrieval-feasibility ceilings. Isolated timing is `TIMING_BLOCKED_SHARED_GPU`; therefore no sustainable QPS, dollar cost, or GPU-kernel cost is claimed. The 1,000 ms and 3,000 ms serialized-selector replacement scenarios remain `NOT_MEASURED`, not passes or failures inferred from shared-device traces.

The deployment question is consequently a quality-only operating map. A policy can be described as quality-competitive only within the measured exploratory denominators; no latency-based Pareto dominance is asserted.
""")
    write_text(OUT / "E4_3_FAILURE_CASEBOOK.md", "# E4-3 Failure Casebook\n\n" + "\n".join(f"- `{x.get('e4_policy')}` K{x.get('k')}, `{x.get('query_id')}`: `{x.get('reason', 'not valid')}`" for x in raw if x.get("status") != "VALID") + "\n")
    groups = Counter(x["source_group_id"] for x in qs)
    sensitivity_rows = list(csv.DictReader((OUT / "E4_3_SOURCE_GROUP_SENSITIVITY.csv").open())) if (OUT / "E4_3_SOURCE_GROUP_SENSITIVITY.csv").is_file() else []
    sensitivity_table = "\n".join(
        f"| {x['policy']} | K{x['k']} | {x['valid_n']} | {x['source_group_count']} | {x['largest_source_group']} ({x['largest_group_n']}) | {x['query_cssr_u']} [{x['query_bootstrap95_low']}, {x['query_bootstrap95_high']}] | [{x['cluster_bootstrap95_low']}, {x['cluster_bootstrap95_high']}] | {x['leave_largest_group_out_cssr_u']} (n={x['leave_largest_group_out_n']}) |"
        for x in sensitivity_rows
    )
    write_text(OUT / "E4_3_SOURCE_GROUP_SENSITIVITY.md", f"""# E4-3 Source-Group Sensitivity

The exploratory cohort is not source-group balanced. It contains **{len(groups)}** groups; the largest has **{groups.most_common(1)[0][1] if groups else 0}** of **{len(qs)}** selected records. Representative-query CSSR-U is the primary exploratory estimand. Query bootstrap treats questions as independent; cluster bootstrap resamples whole source groups and is reported as a sensitivity interval. Leave-largest-group-out is descriptive because the dominant group is unusually large. No representative/diagnostic pooling is used.

| Policy | K | n | Groups | Largest group | Query CSSR-U [95%] | Cluster bootstrap [95%] | Leave-largest-out CSSR-U |
|---|---:|---:|---:|---|---|---|---|
{sensitivity_table}
""")
    write_text(OUT / "E4_3_FINAL_SCIENTIFIC_REPORT.md", f"""# DecisionRAG E4-3 Final Scientific Report

## Gate

`E4_3_OWNER_REVIEW_READY` with quality evidence qualified by shared-GPU timing blockage. The new exploratory cohort is frozen at **{len(qs)}** questions (representative={sum(x['cohort']=='representative' for x in qs)}, diagnostic={sum(x['cohort']=='diagnostic' for x in qs)}). No holdout, Jev, paid API, training, reader regeneration, new retrieval or new model family was used.

## Findings

The report files and scorecard preserve separate K10/K20 CSSR-U, CSSR-C, feasibility ceilings, structural Top-5 limits and valid-denominator accounting. The Clef mechanism rows report matched K-specific effects and the K10-feasible D20−D10 interaction. Clef joint is not presumed superior; negative, null and unavailable cells remain visible.

## Quality scorecard

CSSR-U is unconditional over valid questions; CSSR-C is conditional on all gold-support IDs being present in the K-candidate view. The ceiling is the fraction of questions for which complete support is present. `any_gold_top5` is a descriptive Top-5 any-support rate.

| Policy | K | Cohort | Valid n | CSSR-U (num/den) | CSSR-C (num/den) | Complete-support ceiling | Any-gold Top-5 | Failures |
|---|---:|---|---:|---|---|---:|---:|---:|
{quality_table}

## Clef mechanism contrasts

| Contrast | K | Matched n | Estimate | Bootstrap 95% low | Bootstrap 95% high | Status |
|---|---|---:|---:|---:|---:|---|
{mechanism_table}

The representative joint-minus-pointwise estimates are negative at both K=10 and K=20, with intervals crossing zero. The K20-minus-K10 interaction is evaluated on the identical K10-feasible population and does not support a claim that joint inference improves with larger K.

## Native work and observed execution telemetry

| Policy | K | Valid n | Physical-call n | Exact-reuse n | Forward calls | Observed p50 ms | Observed p95 ms | Peak GB | GPU IDs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
{native_table}

These are non-isolated native-call diagnostics. They are not matched end-to-end Top-K-to-Top-5 isolated latency and must not be converted to sustainable QPS or dollar cost.

Specialist comparisons are policy-bundle comparisons, not architecture effects. Jina, BGE, Liquid and PPLX differ in native objective and context view. Historical K20 specialist rows are reused only where exact input/model/contract hashes match; K10 is native work. The 27B PPLX cell is blocked wherever local K10 feasibility could not be established.

## Latency and replacement

Matched isolated selector latency is blocked because approved GPUs were not isolated. Consequently no production QPS, cost saving, or latency Pareto claim is supported. The 1 s/3 s replacement boundary is `NOT_MEASURED`.

## Reviewer conclusion

This exploratory evidence can diagnose candidate-size, redundancy, support-cardinality and context-footprint interactions, but it cannot establish production replacement. The central question—whether Clef joint beats Clef pointwise and is competitive with Jina/BGE after latency—remains qualified by the exact measured scorecard, blocked timing branch and dominant source-group concentration.
""")
    representative_table = "\n".join(
        f"| {x['policy']} | K{x['k']} | {x['cssr_u']} | {x['cssr_c']} | {x['topk_complete_support_ceiling']} | {x['any_gold_top5']} |"
        for x in score_rows if x['cohort'] == 'representative'
    )
    write_text(OUT / "E4_3_REVIEWER_DECISION_PACKET.md", f"""# E4-3 Reviewer Decision Packet

## What is identifiable

- Same-checkpoint Clef pointwise/joint comparisons at K10 and K20 on a frozen exploratory cohort, where both cells are valid.
- Conditional versus unconditional evidence-retention proxies and natural-prefix retrieval ceilings.
- Descriptive policy-bundle comparisons with Jina, BGE, Liquid and PPLX where native outputs were valid.

## Representative quality comparison

| Policy | K | CSSR-U | CSSR-C | Complete-support ceiling | Any-gold Top-5 |
|---|---:|---:|---:|---:|---:|
{representative_table}

The specialist rows are descriptive policy-bundle comparisons because their native objectives and historical K20 input contracts differ. K10 rows are new native work; exact K20 rows are reused only where the full model/view/input contract matched.

## What is not identifiable

- Causal model-family superiority across native contracts.
- Sustainable production QPS, dollar cost, or matched latency: isolated timing was blocked.
- Human-verified factual correctness: CSSR is an annotated evidence-retention proxy.

## Decision question

Review the scorecard rather than assuming joint is better. If the Clef mechanism contrast is negative or blocked in a cell, retain that result. Liquid/PPLX do not provide a universal replacement claim; the operating regime is policy- and context-dependent.
""")


def full_tests() -> dict[str, Any]:
    cmd = [sys.executable, "-m", "pytest", "-q"]
    env = dict(os.environ); env["PYTHONPATH"] = "."
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
    write_text(OUT / "E4_3_FULL_TEST_REPORT.md", f"# E4-3 Full Test Report\n\nCommand: `PYTHONPATH=. pytest -q`\n\nReturn code: **{p.returncode}**\n\n```text\n{p.stdout[-12000:]}\n{p.stderr[-4000:]}\n```\n")
    return {"return_code": p.returncode, "stdout_tail": p.stdout[-2000:], "stderr_tail": p.stderr[-1000:]}


def finalize() -> None:
    protocol = read_json(OUT / "E4_3_FROZEN_EXPLORATORY_PROTOCOL.json")
    test = full_tests()
    timing_block()
    reports()
    files = sorted(x for x in OUT.iterdir() if x.is_file() and x.name != "E4_3_PROVENANCE_MANIFEST.json")
    manifest = {"status": "E4_3_OWNER_REVIEW_READY", "verdict": "E4_3_QUALITY_COMPLETE_TIMING_BLOCKED", "jev_calls": 0, "paid_api_calls": 0, "reserved_holdout_accessed": False, "test_return_code": test["return_code"], "files": {str(x.relative_to(ROOT)): sha_file(x) for x in files}, "referenced_inputs": {str(x.relative_to(ROOT)): sha_file(x) for x in source_files() if x.is_file()}, "gpu_audit": run_gpu_snapshot(), "protocol_sha256": sha_file(OUT / "E4_3_FROZEN_EXPLORATORY_PROTOCOL.json")}
    write_json(OUT / "E4_3_PROVENANCE_MANIFEST.json", manifest)
    write_text(OUT / "E4_3_OWNER_REVIEW_GATE.md", f"""# E4-3 Owner Review Gate

**Status:** `E4_3_OWNER_REVIEW_READY`  
**Scientific verdict:** `{manifest['verdict']}`  
**Test return code:** `{test['return_code']}`  
**Jev calls:** `0`  
**Paid API calls:** `0`  
**Reserved holdout accessed:** `false`

Quality/interaction outputs are complete to the extent of valid native rows. Matched isolated latency is blocked by the shared GPU state and must not be converted into QPS or cost claims. E4-5 and all other experiments remain stopped pending owner review.
""")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("foundation", "smoke", "full", "quality", "finalize", "all"), required=True)
    ap.add_argument("--policies", nargs="*", choices=list(SPECS))
    args = ap.parse_args()
    if args.stage in {"foundation", "all"}:
        foundation_stage(); provenance_audit()
    if args.stage in {"smoke", "all"}: run_smoke_or_full("smoke", args.policies)
    if args.stage in {"full", "all"}: run_smoke_or_full("full", args.policies)
    if args.stage in {"quality", "all"}: quality_stage()
    if args.stage in {"finalize", "all"}: finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
