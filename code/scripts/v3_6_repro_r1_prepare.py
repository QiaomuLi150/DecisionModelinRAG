#!/usr/bin/env python3
"""Prepare the v3.6 R1 transfer budget grid with the exact frozen builder.

This script is CPU-only and label-blind.  It reads the preserved v2.1 hybrid
Top-20 views, rebuilds the per-candidate prefixes for each length, and writes
an append-only input namespace.  It deliberately does not read qrels.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "scripts" / "v2_1_hybrid_amendment_preflight.py"
BASE = ROOT / "results" / "v2_1_replication"
INPUT = BASE / "stage0_preflight_v2_hybrid" / "INPUT_MANIFEST_V2_HYBRID.jsonl"
OUT = ROOT / "results" / "v3_6_essential" / "followup_v3_repro_fixes" / "R1_transfer_budget_grid"
PPLX_REV = "14c9d90137962d300f3de95633f7b6535e80c5ce"
JINA_REV = "e8a93f33f0b22108f8c2364f8484ce3422552fbc"
PPLX = Path(f"<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub/models--perplexity-ai--pplx-decider-v1.1-27b/snapshots/{PPLX_REV}")
LENGTHS = (2048, 4096, 6144, 8192)
RHO = 128


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canon(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha_obj(obj) -> str:
    return sha_bytes(canon(obj).encode())


def load_rows():
    rows = [json.loads(x) for x in INPUT.open(encoding="utf-8") if x.strip()]
    rows = [x for x in rows if x.get("order_id") == "original"]
    if len(rows) != 2667:
        raise RuntimeError(f"expected 2667 original views, got {len(rows)}")
    ds = {k: sum(1 for x in rows if x.get("dataset") == k) for k in ("enterprise", "musique")}
    if ds != {"enterprise": 500, "musique": 2167}:
        raise RuntimeError(f"unexpected dataset counts: {ds}")
    return rows


def load_frozen_builder():
    spec = importlib.util.spec_from_file_location("frozen_builder", SRC)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    sys.path.insert(0, str(PPLX / "source" / "src"))
    from autojev.model import decision_messages
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(str(PPLX), local_files_only=True)
    return mod, processor, processor.tokenizer, decision_messages


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    builder, processor, tokenizer, decision_messages = load_frozen_builder()
    outputs = {}
    summary = []
    for L in LENGTHS:
        path = OUT / f"prepared_L{L}.jsonl"
        if path.exists():
            raise RuntimeError(f"refusing to overwrite existing file: {path}")
        out_rows = []
        for ix, src in enumerate(rows, 1):
            docs = [{"candidate_id": d["candidate_id"], "content": d["content"]} for d in src["candidates"]]
            # Use the frozen builder's exact serialized request counting and
            # decrement-to-fit rule.  No label or outcome is available here.
            budgeted, info = builder.budget_docs(
                str(src["query"]), docs, processor, decision_messages, tokenizer
            ) if False else (None, None)
            # The historical helper closes over the module's L=8192 constant,
            # so reproduce its exact algorithm with a local target length.
            codes = [f"candidate_{i:03d}" for i in range(20)]
            empty = [{"candidate_id": d["candidate_id"], "content": ""} for d in docs]
            overhead = builder.token_count(processor, decision_messages, builder.make_row(str(src["query"]), empty, codes), codes)
            available = L - RHO - overhead
            budgeted = None
            info = {"overhead_tokens": overhead, "effective_content_budget": available}
            if available >= len(docs):
                initial = max(1, available // len(docs))
                cap = initial
                reductions = 0
                while cap >= 1:
                    candidate_view = [{
                        "candidate_id": d["candidate_id"],
                        "content": builder.prefix(tokenizer, d["content"], cap),
                    } for d in docs]
                    total = builder.token_count(
                        processor, decision_messages,
                        builder.make_row(str(src["query"]), candidate_view, codes), codes
                    )
                    if total <= L - RHO:
                        budgeted = candidate_view
                        info.update({
                            "initial_per_passage_cap": initial,
                            "final_per_passage_cap": cap,
                            "deterministic_cap_reductions": reductions,
                            "serialized_tokens": total,
                            "builder_status": "PASS",
                        })
                        break
                    cap -= 1
                    reductions += 1
            if budgeted is None:
                info.update({"builder_status": "FAIL", "error_class": "UNBUDGETABLE_QUERY_OR_OVERHEAD"})
            out_rows.append({
                "dataset": src["dataset"],
                "canonical_id": src["canonical_id"],
                "query_id": src["query_id"],
                "group_id": src.get("group_id", ""),
                "query": src["query"],
                "candidate_ids": [d["candidate_id"] for d in docs],
                "source_slate_sha256": src["source_slate_sha256"],
                "candidate_content_sha256": [sha_bytes(d["content"].encode()) for d in docs],
                "candidates": docs,
                "budgeted_candidates": budgeted or [],
                "budget_info": info,
                "serialized_tokens": info.get("serialized_tokens"),
                "length_limit": L,
                "safety_margin_tokens": RHO,
                "label_blind": True,
            })
            if ix % 250 == 0:
                print(json.dumps({"L": L, "completed": ix, "total": len(rows)}), flush=True)
        path.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in out_rows), encoding="utf-8")
        outputs[L] = path
        built = [x for x in out_rows if x["budget_info"].get("builder_status") == "PASS"]
        failures = [x for x in out_rows if x["budget_info"].get("builder_status") != "PASS"]
        bad = [x for x in built if int(x["serialized_tokens"]) > L - RHO]
        if bad:
            raise RuntimeError(f"serialized limit violation at L={L}: {len(bad)}")
        summary.append({
            "dataset": "all",
            "L": L,
            "n": len(out_rows),
            "builder_pass_n": len(built),
            "builder_failure_n": len(failures),
            "mean_serialized_tokens": sum(x["serialized_tokens"] for x in built) / len(built) if built else None,
            "max_serialized_tokens": max((x["serialized_tokens"] for x in built), default=None),
            "limit_minus_max": (L - RHO - max(x["serialized_tokens"] for x in built)) if built else None,
            "manifest_sha256": sha_file(path),
        })
        for dataset in ("enterprise", "musique"):
            sub = [x for x in out_rows if x["dataset"] == dataset]
            bp = [x for x in sub if x["budget_info"].get("builder_status") == "PASS"]
            summary.append({
                "dataset": dataset,
                "L": L,
                "n": len(sub),
                "builder_pass_n": len(bp),
                "builder_failure_n": len(sub) - len(bp),
                "mean_serialized_tokens": sum(x["serialized_tokens"] for x in bp) / len(bp) if bp else None,
                "max_serialized_tokens": max((x["serialized_tokens"] for x in bp), default=None),
                "limit_minus_max": (L - RHO - max(x["serialized_tokens"] for x in bp)) if bp else None,
            })
    (OUT / "BUILDER_SUMMARY.json").write_text(json.dumps({"status": "PASS", "lengths": summary, "source_manifest": str(INPUT.relative_to(ROOT)), "source_manifest_sha256": sha_file(INPUT), "labels_included": False}, indent=2) + "\n")
    protocol = {
        "protocol_id": "decisionrag-v3.6-repro-fixes-r1-r2",
        "version": "R1-R2-1.0",
        "scope": "Two reproducibility fixes; no new benchmark or model family",
        "r1": {
            "population": {"enterprise": 500, "musique": 2167, "total": 2667},
            "lengths": list(LENGTHS),
            "safety_margin_tokens": RHO,
            "builder": "measure exact empty-candidate serialized request overhead, then decrement one shared per-candidate prefix cap until exact serialized request <= L-128",
            "tokenizer_revision": PPLX_REV,
            "models": {"pplx": PPLX_REV, "jina": JINA_REV},
            "no_retries": True,
            "labels_joined_after_raw_freeze": True,
            "bootstrap_draws": 40000,
            "bootstrap_seed": 731,
            "bootstrap_unit": "question",
            "failure_score": 0.0,
            "metrics": ["nDCG@5", "FS@5"],
        },
        "r2": {
            "population": 1384,
            "comparison": "v3.6 sweep jina_two_stage_8192__8192_reuse vs v1.5 jina_two_stage_pplxtext__original",
            "trace_comparison": "stage request hashes, candidate membership/order, score vectors, survivor IDs, reconstructed window content hashes",
            "rerun_if_input_mismatch": True,
        },
        "source_manifest": str(INPUT.relative_to(ROOT)),
        "source_manifest_sha256": sha_file(INPUT),
        "prepared_files": {str(L): {"path": str(outputs[L].relative_to(ROOT)), "sha256": sha_file(outputs[L])} for L in LENGTHS},
        "labels_included": False,
    }
    proto = OUT.parent / "PROTOCOL.json"
    if proto.exists():
        raise RuntimeError(f"refusing to overwrite existing protocol: {proto}")
    proto.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    (OUT.parent / "PROTOCOL.sha256").write_text(json.dumps({"path": str(proto.relative_to(ROOT)), "sha256": sha_file(proto), "frozen_before_inference": True}, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "prepared": {str(k): str(v.relative_to(ROOT)) for k, v in outputs.items()}, "protocol_sha256": sha_file(proto)}, indent=2))


if __name__ == "__main__":
    main()
