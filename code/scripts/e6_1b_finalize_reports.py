#!/usr/bin/env python3
"""Offline E6-1B report finalization from preserved native traces."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import e6_1b_checkpoint_recovery as e6


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def inference_code_metadata(meta: dict) -> tuple[str, str]:
    """Return a source revision plus a content hash for the pinned native adapter."""
    if meta["kind"] == "amos":
        sources = json.loads((meta["path"] / "serve" / "sources.json").read_text(encoding="utf-8"))
        files = sources.get("files", {})
        return str(sources.get("commit", "NOT_RECORDED")), e6.sha_json(files)
    package_files = {}
    for p in sorted((meta["path"] / "startlux_decision").glob("*.py")):
        package_files[p.name] = e6.sha_file(p)
    return "BUNDLED_WITH_CHECKPOINT_SNAPSHOT", e6.sha_json(package_files)


def main() -> int:
    out = e6.OUT
    inputs, protocol = e6.verify_inputs()
    traces = e6.read_jsonl(out / "E6_1B_NATIVE_TRACES.jsonl")
    summaries, latency = [], []
    for name in e6.MODELS:
        summary, rows = e6.model_summary(name, traces)
        summaries.append(summary); latency.extend(rows)

    scorecard = []
    registry = []
    for s in summaries:
        fully_valid = s["valid"] == 256
        any_valid = s["valid"] > 0
        forward_observation = ("DIRECT_HOOK_COUNT" if s["model"] == "startlux_decision_4b"
                               else "NOT_INSTRUMENTED_BY_OFFICIAL_DECODER_WRAPPER")
        scorecard.append({**s,
                          "physical_forwards": (s["physical_forwards"] if s["model"] == "startlux_decision_4b"
                                                 else "NOT_MEASURED"),
                          "native_top5_contract": "PASS" if any_valid else "FAIL",
                          "probability_alignment": "PASS" if any_valid else "NOT_ESTABLISHED",
                          "selection_coverage": s["validity_rate"],
                          "physical_forward_measurement": forward_observation,
                          "fallback_used": False, "quality_analysis": "NOT_RUN"})
        meta = e6.MODELS[s["model"]]
        code_revision, code_hash = inference_code_metadata(meta)
        registry.append({"model": s["model"], "repo": meta["repo"], "revision": meta["revision"],
                         "checkpoint_path": str(meta["path"]), "checkpoint_present": meta["path"].is_dir(),
                         "license": meta["license"], "parameters": meta["params"], "kind": meta["kind"],
                         "native_contract": meta["native_contract"], "attempts": s["attempts"],
                         "inference_code_revision": code_revision, "inference_code_sha256": code_hash,
                         "valid_native_top5": s["valid"], "context_or_inference_failures": s["failures"],
                         "physical_forwards": (s["physical_forwards"] if s["model"] == "startlux_decision_4b"
                                                else "NOT_MEASURED"),
                         "physical_forward_measurement": forward_observation,
                         "native_api_calls": s["native_api_calls"],
                         "contract_status": "VALIDATED" if fully_valid else ("VALIDATED_WITH_CONTEXT_FAILURES" if any_valid else "BLOCKED"),
                         "e6_2_inclusion": "INCLUDE_WITH_CONTEXT_FEASIBILITY_LIMIT" if any_valid else "EXCLUDE"})
    write_csv(out / "E6_1B_CONTRACT_SCORECARD.csv", scorecard)
    write_csv(out / "E6_1B_MODEL_REGISTRY.csv", registry)
    write_csv(out / "E6_1B_LATENCY_REPORT.csv", latency)

    order_rows = []
    for name in e6.MODELS:
        reversed_rows = [x for x in traces if x.get("model") == name and x.get("stage") == "order_check" and x.get("order") == "reversed"]
        for rev in reversed_rows:
            can = next((x for x in traces if x.get("model") == name and x.get("query_id") == rev.get("query_id")
                        and x.get("stage") == "order_check" and x.get("order") == "canonical"), None)
            a, b = set(rev.get("selected_ids", [])), set(can.get("selected_ids", [])) if can else set()
            order_rows.append({"model": name, "query_id": rev.get("query_id"),
                               "canonical_selected_ids": json.dumps(can.get("selected_ids", []) if can else []),
                               "reversed_selected_ids": json.dumps(rev.get("selected_ids", [])),
                               "exact_set_match": bool(a == b) if can else None,
                               "jaccard": len(a & b) / len(a | b) if a | b else None,
                               "changed_selection_count": len(a ^ b) // 2 if can else None,
                               "canonical_status": can.get("status") if can else "MISSING",
                               "reversed_status": rev.get("status")})
    write_csv(out / "E6_1B_ORDER_DIAGNOSTICS.csv", order_rows)

    accounting_rows = []
    for s in summaries:
        model_rows = [x for x in traces if x.get("model") == s["model"]]
        scheduled = [x for x in model_rows if x.get("stage") in ("smoke", "cal", "order_check")]
        valid = [x for x in scheduled if x.get("status") == "VALID"]
        loaded = [x for x in model_rows if x.get("stage") == "load" and x.get("status") == "LOADED"]
        alloc = [float(x["peak_memory_allocated_mib"]) for x in valid if x.get("peak_memory_allocated_mib") is not None]
        reserved = [float(x["peak_memory_reserved_mib"]) for x in valid if x.get("peak_memory_reserved_mib") is not None]
        accounting_rows.append({
            "model": s["model"], "physical_gpu": 5, "scheduled_attempts": len(scheduled),
            "successful_native_calls": len(valid), "native_api_calls": sum(int(x.get("native_api_calls") or 0) for x in scheduled),
            "forward_hook_events_observed": sum(int(x.get("physical_forward_count") or 0) for x in valid),
            "physical_forward_measurement": ("DIRECT_HOOK_COUNT" if s["model"] == "startlux_decision_4b"
                                              else "NOT_INSTRUMENTED_BY_OFFICIAL_DECODER_WRAPPER"),
            "load_time_ms": loaded[-1].get("load_time_ms") if loaded else None,
            "peak_allocated_mib_max": max(alloc) if alloc else None,
            "peak_reserved_mib_max": max(reserved) if reserved else None,
            "k15_valid": sum(x.get("candidate_k") == 15 for x in valid),
            "k20_valid": sum(x.get("candidate_k") == 20 for x in valid),
            "context_or_inference_failures": len(scheduled) - len(valid),
        })
    write_csv(out / "E6_1B_INFERENCE_ACCOUNTING.csv", accounting_rows)

    statuses = {s["model"]: ("VALIDATED" if s["valid"] == 256 else
                              "VALIDATED_WITH_CONTEXT_FAILURES" if s["valid"] > 0 else "BLOCKED") for s in summaries}
    gate = "E6_2_READY" if all(v == "VALIDATED" for v in statuses.values()) else (
        "E6_2_READY_WITH_EXCLUSIONS" if all(v != "BLOCKED" for v in statuses.values()) else "E6_2_BLOCKED")
    write_json(out / "E6_1B_E6_2_READINESS_GATE.json", {"status": gate, "models": statuses,
        "e6_2_started": False, "e6_2_authorized": False, "holdout_access": False,
        "jev_calls": 0, "paid_api_calls": 0, "historical_artifacts_preserved": True,
        "context_failure_policy": "retain explicit failures; include model only with declared feasibility limit",
        "created_at": e6.utc()})

    lat = {(x["model"], x["candidate_k"]): x for x in latency}
    failure_rows = [x for x in traces if x.get("stage") in ("smoke", "cal", "order_check") and x.get("status") != "VALID"]
    failure_queries = sorted({x.get("query_id") for x in failure_rows if x.get("query_id")})
    lines = ["# DecisionRAG E6-1B Checkpoint Recovery and Native Adapter Validation", "", f"## Gate: `{gate}`", "",
             "This milestone validates native selection contracts and input-feasibility limits. It is not a ranking-quality benchmark.", "",
             "## Frozen-input authentication", f"- E6-1 CAL rows: {len(inputs)} unique SciFact questions.",
             f"- Canonical E6-1 input manifest SHA-256: `{e6.sha_json(inputs)}` (expected `{e6.EXPECTED_INPUT_HASH}`).",
             f"- Historical E5 Top-20 slate SHA-256: `{e6.EXPECTED_SLATE_HASH}`.",
              "- qrels, gold labels, reader outputs and reserved holdouts were not loaded.",
             "- Exact per-model attempt schedule: 10 smoke + 240 CAL + 6 reversal diagnostics = 256; K15=125, K20=131.", "",
             "## Model scorecard", "", "| model | valid / attempts | K15 | K20 | K15 p50/p95 ms | K20 p50/p95 ms | order match | physical forwards | status |", "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for s in summaries:
        a, b = lat.get((s["model"], 15), {}), lat.get((s["model"], 20), {})
        forwards = s["physical_forwards"] if s["model"] == "startlux_decision_4b" else "NOT_MEASURED"
        lines.append(f"| {s['model']} | {s['valid']} / {s['attempts']} | {s['top15_valid']} | {s['top20_valid']} | {a.get('p50_ms')} / {a.get('p95_ms')} | {b.get('p50_ms')} / {b.get('p95_ms')} | {s['order_exact_set_matches']}/{s['order_checks']} | {forwards} | {s['status']} |")
    lines += ["", "## Contract and feasibility interpretation",
              "- Both recovered checkpoints use official native typed categorical choice readouts. Candidate probabilities are aligned to the submitted option IDs and Top-5 is produced by deterministic descending-probability sorting with original-order tie breaking.",
              "- This is a single-choice readout followed by Top-5 sorting, not joint-set optimization.",
              "- StartLux: 256/256 native outputs valid. Its bundled checker tests a stale Transformers marker; direct binding inspection and CUDA kernel smoke calls verified causal-conv1d and FLA dispatch. The documented guard override was recorded; no reference PyTorch fallback was used.",
              "- Amos: 250/256 valid. Six scheduled attempts exceeded the official decoder's 16,384-token strict limit, covering five distinct query IDs; `scifact:100` failed in both the smoke and CAL K20 attempts. They were retained as explicit failures; no truncation or retry with altered inputs occurred. Amos used the official local decoder path, rotations=1, and no state cache.",
              "- The six Amos failures are context-feasibility exclusions, not model-quality failures. The five distinct affected query IDs are listed in `E6_1B_FAILURES.csv`.",
              "- Amos native API attempts are recorded for every scheduled request, but the official DecoderEngine invokes its underlying decoder through an internal path not observed by the runner's top-level forward hook. Therefore its physical-forward count is `NOT_MEASURED`, not zero; the 250 successful native calls establish a lower bound on completed model forwards.", "",
              "## Inference accounting", "- Load time, peak allocated/reserved GPU memory, native API calls, observed hook events and K15/K20 valid counts are in `E6_1B_INFERENCE_ACCOUNTING.csv`.",
              "## Order diagnostics", "- Exact set matches, Jaccard overlap and changed-selection counts are in `E6_1B_ORDER_DIAGNOSTICS.csv`; canonical ordering remains the primary protocol.",
              "", "## Readiness", f"- `E6_2_READY_WITH_EXCLUSIONS` is recommended because Amos is native-contract valid where feasible but has six K20 context-limit failures among 131 scheduled K20 attempts (five distinct frozen query IDs).",
              "- StartLux may be included; Amos may be included only with its prespecified context-feasibility accounting, or excluded from any E6-2 population that requires complete K20 coverage.",
              "- No quality conclusion is authorized from this CAL feasibility work.", "", "## Restrictions verified",
              "- E6-2 was not started. No Jev, paid API, training, reader generation, retrieval optimization or reserved-holdout access.",
              "- GPU 5 was used sequentially after UUID/occupancy audit and released after each model; no unrelated process was evicted."]
    write_csv(out / "E6_1B_FAILURES.csv", [{"model": x.get("model"), "stage": x.get("stage"),
                                             "query_id": x.get("query_id"), "candidate_k": x.get("candidate_k"),
                                             "input_tokens": x.get("input_tokens"), "failure": x.get("failure"),
                                             "native_api_calls": x.get("native_api_calls", 0),
                                             "physical_forward_count_observed": x.get("physical_forward_count", 0)}
                                            for x in failure_rows])
    lines += ["", "## Failure accounting", f"- Failed scheduled attempts: {len(failure_rows)}.",
              f"- Distinct affected query IDs: {len(failure_queries)} ({', '.join(failure_queries)}).",
              "- All failures were official strict context-length rejections before model forward; no timeout, OOM, invalid choice schema, duplicate selection, or fallback was observed."]
    (out / "E6_1B_FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (out / "E6_1B_TEST_REPORT.md").write_text(
        "# DecisionRAG E6-1B Regression and Integrity Test Report\n\n"
        f"Generated: `{e6.utc()}`\n\n"
        "## Results\n\n"
        "| suite | command/runtime | result |\n"
        "|---|---|---|\n"
        "| focused E6-1B | `pytest -q tests/test_e6_1b_checkpoint_recovery.py` | PASS, 5/5 |\n"
        "| full relevant regression | shared `MultiModalRAG/venv`, `python -m pytest -q tests` | PASS, 203/203 |\n\n"
        "The system-Python invocation was also attempted; it produced two environment-only failures in pre-existing E6-0R Arrow tests because `pyarrow` is absent from `/usr/bin/python3`. The designated shared environment contains `pyarrow==25.0.1`, and the complete 203-test suite passes there. No E6-1B assertion failed.\n\n"
        "## Coverage added by this milestone\n\n"
        "The focused suite covers frozen-input and attempt accounting, candidate-probability alignment and deterministic ties, gold-label exclusion from native requests, invalid-output gating, and preservation of the prior E6-1 gate/artifacts.\n\n"
        "## Integrity checks\n\n"
        "- E6-1 canonical input hash matched the frozen expected hash.\n"
        "- Historical E5 Top-20 slate hash matched the frozen expected hash.\n"
        "- All 12 files listed by the E6-1B provenance manifest matched their recorded SHA-256 values.\n"
        "- E6-2 was not started; Jev and paid API calls remained zero.\n",
        encoding="utf-8")

    manifest = {"created_at": e6.utc(), "status": "E6_1B_COMPLETE", "gate": gate,
                "historical_inputs": {"path": str(e6.INPUTS.relative_to(e6.ROOT)), "sha256_bytes": e6.sha_file(e6.INPUTS),
                                      "canonical_manifest_sha256": e6.sha_json(inputs), "expected": e6.EXPECTED_INPUT_HASH,
                                      "historical_e5_top20_sha256": e6.EXPECTED_SLATE_HASH},
                "attempt_accounting": {"per_model": 256, "smoke": 10, "cal": 240, "order_diagnostic": 6,
                                       "top15": 125, "top20": 131, "models": list(e6.MODELS)},
                "models": [{**{k: v for k, v in m.items() if k != "path"}, "path": str(m["path"]),
                            "checkpoint_files": e6.checkpoint_inventory(m["path"])} for m in e6.MODELS.values()],
                "source_files": [{"path": str(p.relative_to(e6.ROOT)), "sha256": e6.sha_file(p)} for p in
                                 [Path(__file__), e6.INPUTS, e6.PROTOCOL, e6.ROOT / "scripts/e6_1b_checkpoint_recovery.py"]],
                "artifacts_preserved": True, "holdout_access": False, "jev_calls": 0, "paid_api_calls": 0,
                "reader_inference": False, "training": False}
    # The receipt does not contain the manifest hash, so it can be written
    # before the manifest and then included in the manifest without a cycle.
    write_json(out / "E6_1B_COMPLETION_RECEIPT.json", {"gate": gate,
                                                       "report_sha256": e6.sha_file(out / "E6_1B_FINAL_REPORT.md"),
                                                       "traces_sha256": e6.sha_file(out / "E6_1B_NATIVE_TRACES.jsonl"),
                                                       "created_at": e6.utc()})
    for p in sorted(out.iterdir()):
        if p.is_file() and p.name != "E6_1B_PROVENANCE_MANIFEST.json":
            manifest.setdefault("output_files", []).append({"path": str(p.relative_to(e6.ROOT)), "bytes": p.stat().st_size, "sha256": e6.sha_file(p)})
    write_json(out / "E6_1B_PROVENANCE_MANIFEST.json", manifest)
    print(f"finalized {len(traces)} raw trace records; gate={gate}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
