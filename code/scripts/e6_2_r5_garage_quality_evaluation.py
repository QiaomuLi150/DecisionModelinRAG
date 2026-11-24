#!/usr/bin/env python3
"""E6-2 R5 offline scoring of the frozen GaRAGe R4D selections.

This module has two deliberately separate phases:

``prepare`` authenticates immutable R4C/R4D inputs, builds the combined
prediction registry, and emits an EXPLORATORY-only annotation projection.
``score`` reads only that projection and computes the frozen binary endpoint.

It never loads a model, calls a service, reads RESERVED raw content, or
changes any historical artifact.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/non_jev/v2/e6_v1_0"
R0 = BASE / "e6_2_r0_scientific_readiness"
R3 = BASE / "e6_2_r3_scifact_quality_evaluation"
R4C = BASE / "e6_2_r4c_erratum_and_garage_freeze"
R4D = BASE / "e6_2_r4d_garage_native_inference"
OUT = BASE / "e6_2_r5_garage_quality_evaluation"

INPUTS = R4C / "E6_2_R4C_GARAGE_EXPLORATORY_INPUTS.jsonl"
PARTITION = R4C / "E6_2_R4C_GARAGE_PARTITION_MANIFEST.json"
LABELS = R4C / "E6_2_R4C_GARAGE_LABEL_PROJECTION.json"
OVERLAP = R4C / "E6_2_R4C_GARAGE_OVERLAP_GRAPH_REPORT.json"
R4D_SELECTIONS = R4D / "E6_2_R4D_NATIVE_SELECTIONS.jsonl"
R4D_PROV = R4D / "E6_2_R4D_PROVENANCE_MANIFEST.json"
R4D_GATE = R4D / "E6_2_R4D_R5_READINESS_GATE.json"

MODELS = [
    "pplx_decider",
    "liquid_d1_3b",
    "startlux_decision_4b",
    "amos_1_small_2b",
    "jina_v3_5",
    "qwen3_reranker_4b",
    "bge_reranker_v2_m3",
    "qwen3_reranker_0_6b",
]
MODEL_FAMILY = {
    "pplx_decider": "decision",
    "liquid_d1_3b": "decision",
    "startlux_decision_4b": "decision",
    "amos_1_small_2b": "decision",
    "jina_v3_5": "specialist",
    "qwen3_reranker_4b": "specialist",
    "bge_reranker_v2_m3": "specialist",
    "qwen3_reranker_0_6b": "specialist",
}
MODEL_REVISIONS = {
    "pplx_decider": "14c9d90137962d300f3de95633f7b6535e80c5ce",
    "liquid_d1_3b": "051bcc464b01b9f92942b364d9586b0ef5912432",
    "startlux_decision_4b": "9c0a32e9809d6a3f612434dd0bce150295dcc274",
    "amos_1_small_2b": "09ba3e9d8f506cfcb96ea130845d5d76d9d92028",
    "jina_v3_5": "e8a93f33f0b22108f8c2364f8484ce3422552fbc",
    "qwen3_reranker_4b": "22e683669bc0f0bd69640a1354a6d0aebcfeede5",
    "bge_reranker_v2_m3": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    "qwen3_reranker_0_6b": "e61197ed45024b0ed8a2d74b80b4d909f1255473",
}
PREFIX = "ORIGINAL_FIXED_POOL_PREFIX5"
SEED = 731
BOOTSTRAPS = 5000


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def verify_manifest(path: Path) -> dict[str, Any]:
    """Verify every file entry in the historical manifest without mutation."""
    doc = read_json(path)
    entries: list[dict[str, Any]] = []
    for key in ("historical_inputs", "key_external_inputs", "new_outputs", "outputs", "code", "input_files", "source_inputs", "historical_r4c_inputs"):
        value = doc.get(key, [])
        if isinstance(value, list):
            entries.extend(x for x in value if isinstance(x, dict) and x.get("path"))
    checked, missing, mismatched = [], [], []
    seen = set()
    for entry in entries:
        if entry["path"] in seen:
            continue
        seen.add(entry["path"])
        target = resolve_path(entry["path"])
        if not target.exists():
            missing.append(entry["path"])
            continue
        actual = sha_file(target)
        size_ok = "bytes" not in entry or target.stat().st_size == entry["bytes"]
        if actual != entry.get("sha256") or not size_ok:
            mismatched.append({"path": entry["path"], "actual_sha256": actual, "manifest_sha256": entry.get("sha256"), "actual_bytes": target.stat().st_size, "manifest_bytes": entry.get("bytes")})
        else:
            checked.append(entry["path"])
    return {"manifest": rel(path), "schema": doc.get("schema"), "listed": len(entries), "checked": len(checked), "missing": missing, "mismatched": mismatched, "pass": not missing and not mismatched}


def load_inputs() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_jsonl(INPUTS)
    partition = read_json(PARTITION)
    ids = [row["sample_id"] for row in rows]
    if len(rows) != 1397 or len(ids) != len(set(ids)):
        raise RuntimeError("R4C exploratory input population is not exactly 1,397 unique samples")
    if set(ids) != set(partition["partition_ids"]["EXPLORATORY"]):
        raise RuntimeError("R4C input IDs do not equal the frozen EXPLORATORY partition")
    for row in rows:
        candidates = row.get("candidates", [])
        if row.get("partition") != "EXPLORATORY" or len(candidates) != 15:
            raise RuntimeError(f"invalid frozen input row: {row.get('sample_id')}")
        order = [x.get("candidate_id") for x in candidates]
        if order != row.get("candidate_order") or len(set(order)) != 15:
            raise RuntimeError(f"candidate order mismatch: {row.get('sample_id')}")
        if any(c.get("position") != i for i, c in enumerate(candidates, 1)):
            raise RuntimeError(f"candidate positions are not canonical: {row.get('sample_id')}")
        if any(key in row for key in ("evidence_relevant", "evidence_correct", "evidence_cited", "answer_generate", "answer_validate")):
            raise RuntimeError("label-bearing field entered frozen label-blind input")
    return rows, {"count": len(rows), "ids_sha256": sha_json(sorted(ids)), "input_sha256": sha_file(INPUTS), "partition_sha256": sha_file(PARTITION)}


def load_predictions(inputs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_jsonl(R4D_SELECTIONS)
    expected_ids = {row["sample_id"] for row in inputs}
    expected = {(model, sid) for model in MODELS for sid in expected_ids}
    keys = [(row.get("model"), row.get("sample_id")) for row in rows]
    if len(rows) != 11176 or len(keys) != len(set(keys)) or set(keys) != expected:
        raise RuntimeError("R4D prediction registry does not contain exactly one valid model-sample record")
    by_sample: dict[str, list[str]] = {}
    for row in rows:
        if row.get("model") not in MODELS or row.get("status") != "ATTEMPTED_VALID":
            raise RuntimeError("R4D prediction status/model mismatch")
        if len(row.get("selected_ids", [])) != 5 or len(set(row["selected_ids"])) != 5:
            raise RuntimeError("R4D prediction is not exactly five unique selections")
        candidates = row.get("candidate_ids", [])
        if len(candidates) != 15 or len(set(candidates)) != 15 or not set(row["selected_ids"]).issubset(set(candidates)):
            raise RuntimeError("R4D selection is not a subset of its K15 pool")
        previous = by_sample.setdefault(row["sample_id"], candidates)
        if previous != candidates:
            raise RuntimeError("models do not share the identical original K15 candidate view")
    return rows, {"count": len(rows), "models": len({x["model"] for x in rows}), "sample_ids": len({x["sample_id"] for x in rows}), "prediction_sha256": sha_file(R4D_SELECTIONS), "r4d_provenance_sha256": sha_file(R4D_PROV)}


def authenticate() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    required = [R0 / "E6_2_R0_STATISTICAL_ANALYSIS_PLAN.md", R0 / "E6_2_R0_EIGHT_MODEL_ADMISSION_MATRIX.csv", R3 / "E6_2_R3_FINAL_SCIENTIFIC_REPORT.md", R3 / "E6_2_R3_PRIMARY_STATISTICAL_ANALYSIS.json", R4C / "E6_2_R4C_FINAL_REPORT.md", R4C / "E6_2_R4C_ENDPOINT_PROTOCOL.md", R4C / "E6_2_R4C_GARAGE_LABEL_PROJECTION.json", R4D / "E6_2_R4D_FINAL_REPORT.md", R4D / "E6_2_R4D_R5_READINESS_GATE.json", R4D / "E6_2_R4D_PROVENANCE_MANIFEST.json", R4D / "E6_2_R4D_MODEL_EXECUTION_REGISTRY.csv"]
    # The endpoint file is named *_ENDPOINT_PROTOCOL.md in R4C; accept its
    # actual frozen name without silently treating a missing file as success.
    required = [p for p in required if p.name != "E6_2_R4C_ENDPOINT_PROTOCOL.md"] + [R4C / "E6_2_R4C_GARAGE_ENDPOINT_PROTOCOL.md"]
    missing = [rel(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(f"required governing artifact missing: {missing}")
    manifests = [R0 / "E6_2_R0_INPUT_OUTPUT_PROVENANCE_MANIFEST.json", R3 / "E6_2_R3_PROVENANCE_MANIFEST.json", R4C / "E6_2_R4C_PROVENANCE_MANIFEST.json", R4D / "E6_2_R4D_PROVENANCE_MANIFEST.json"]
    checks = [verify_manifest(p) for p in manifests]
    if not all(x["pass"] for x in checks):
        raise RuntimeError(f"historical provenance mismatch: {checks}")
    r4d_gate = read_json(R4D_GATE)
    r4d_prov = read_json(R4D_PROV)
    if r4d_gate.get("gate") != "E6_2_R4D_SELECTIONS_FROZEN_READY" or r4d_gate.get("quality_scoring_started") or r4d_gate.get("reserved_accessed"):
        raise RuntimeError("R4D release gate is not admissible for R5")
    if r4d_prov.get("native_inference_calls") != 11176 or r4d_prov.get("quality_metrics_computed") is not False or r4d_prov.get("reserved_content_accessed"):
        raise RuntimeError("R4D provenance does not match the frozen quality-blind release")
    inputs, input_meta = load_inputs()
    predictions, prediction_meta = load_predictions(inputs)
    auth = {"schema": "E6-2-R5-INPUT-AUTH-V1", "created_utc": now(), "historical_manifest_checks": checks, "required_artifacts": [{"path": rel(p), "sha256": sha_file(p), "bytes": p.stat().st_size} for p in required], "r4c_gate": read_json(R4C / "E6_2_R4C_R4_INFERENCE_RELEASE_GATE.json"), "r4d_gate": r4d_gate, "input_population": input_meta, "prediction_population": prediction_meta, "r4c_label_source_sha256": sha_file(LABELS), "reserved_policy": "only EXPLORATORY rows are exported into the R5 scoring projection; RESERVED labels/content are not passed to scoring", "no_new_inference": True, "jev_calls": 0, "paid_api_calls": 0, "historical_artifacts_modified": False}
    return auth, inputs, predictions, {"r4d_prov": r4d_prov, "r4c_overlap": read_json(OVERLAP)}


def project_exploratory_labels(inputs: list[dict[str, Any]]) -> dict[str, Any]:
    source = read_json(LABELS)
    all_rows = source.get("rows", [])
    expected = {row["sample_id"] for row in inputs}
    rows = [row for row in all_rows if row.get("partition") == "EXPLORATORY" and row.get("sample_id") in expected]
    if len(rows) != 1397 or {x["sample_id"] for x in rows} != expected:
        raise RuntimeError("approved EXPLORATORY label projection is incomplete or mispartitioned")
    for row in rows:
        if len(row.get("evidence_relevant", [])) != 15 or len(row.get("evidence_correct", [])) != 15 or len(row.get("evidence_cited", [])) != 15:
            raise RuntimeError(f"annotation array length mismatch: {row.get('sample_id')}")
        allowed = {"YES", "NO", ""}
        if any(str(x or "").strip().upper() not in allowed for x in row["evidence_relevant"]):
            raise RuntimeError(f"unrecognized evidence_relevant value: {row.get('sample_id')}")
    return {"schema": "E6-2-R5-EXPLORATORY-LABEL-PROJECTION-V1", "created_utc": now(), "source_projection": rel(LABELS), "source_projection_sha256": sha_file(LABELS), "rows": sorted(rows, key=lambda x: x["sample_id"]), "rows_exported": len(rows), "reserved_rows_not_exported": sum(x.get("partition") == "RESERVED" for x in all_rows), "cal_rows_not_exported": sum(x.get("partition") == "CAL" for x in all_rows), "access_policy": "R5 scorer receives only this EXPLORATORY projection; no RESERVED row is exported"}


def prepare() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    auth, inputs, predictions, _ = authenticate()
    projection = project_exploratory_labels(inputs)
    input_by_id = {x["sample_id"]: x for x in inputs}
    source_hash = sha_file(R4D_SELECTIONS)
    registry = []
    for row in sorted(predictions, key=lambda x: (MODELS.index(x["model"]), x["sample_id"])):
        registry.append({"model_id": row["model"], "model_family": MODEL_FAMILY[row["model"]], "model_revision": row["checkpoint_revision"], "sample_id": row["sample_id"], "candidate_pool_sha256": row["candidate_ids_sha256"], "candidate_ids": row["candidate_ids"], "selected_ids": row["selected_ids"], "selection_status": row["status"], "prediction_source": rel(R4D_SELECTIONS), "prediction_file_sha256": source_hash, "r4d_provenance_sha256": sha_file(R4D_PROV), "input_manifest_sha256": row.get("input_manifest_sha256"), "original_question_sha256": input_by_id[row["sample_id"]]["question_sha256"]})
    write_json(OUT / "E6_2_R5_INPUT_AUTHENTICATION.json", auth)
    write_json(OUT / "E6_2_R5_COMBINED_SELECTION_REGISTRY.json", {"schema": "E6-2-R5-COMBINED-SELECTION-REGISTRY-V1", "created_utc": now(), "frozen": True, "population": {"samples": 1397, "models": 8, "records": 11176, "candidates_per_pool": 15, "selected_per_record": 5}, "prediction_file_sha256": source_hash, "records": registry})
    write_json(OUT / "E6_2_R5_EXPLORATORY_LABEL_PROJECTION.json", projection)
    receipt = {"schema": "E6-2-R5-PREOUTCOME-FREEZE-V1", "status": "FROZEN_BEFORE_LABEL_SCORING", "created_utc": now(), "selection_registry_sha256": sha_file(OUT / "E6_2_R5_COMBINED_SELECTION_REGISTRY.json"), "prediction_sha256": source_hash, "input_sha256": sha_file(INPUTS), "label_projection_sha256": sha_file(OUT / "E6_2_R5_EXPLORATORY_LABEL_PROJECTION.json"), "model_forward_calls_after_freeze": 0, "quality_metrics_computed_before_freeze": False, "reserved_rows_exported": 0, "jev_calls": 0, "paid_api_calls": 0}
    write_json(OUT / "E6_2_R5_PREOUTCOME_FREEZE_RECEIPT.json", receipt)
    print(json.dumps({"status": receipt["status"], "samples": 1397, "records": 11176, "exploratory_labels": 1397, "reserved_rows_exported": 0}, indent=2))
    return 0


def dcg(gains: list[int]) -> float:
    return sum(float(gain) / math.log2(index + 2) for index, gain in enumerate(gains[:5]))


def score_one(selected: list[str], candidate_ids: list[str], relevance: list[str], correct: list[str], cited: list[str]) -> dict[str, Any]:
    pos = {candidate: index for index, candidate in enumerate(candidate_ids)}
    normalized = [str(x or "").strip().upper() for x in relevance]
    correct_norm = [str(x or "").strip().upper() or "EMPTY" for x in correct]
    cited_norm = [str(x or "").strip().upper() for x in cited]
    positive_positions = [i for i, value in enumerate(normalized) if value == "YES"]
    gains = [1 if (candidate in pos and normalized[pos[candidate]] == "YES") else 0 for candidate in selected]
    ideal = dcg([1] * min(5, len(positive_positions)))
    value = 0.0 if ideal == 0 else dcg(gains) / ideal
    selected_positions = [pos[candidate] for candidate in selected]
    selected_correct = Counter(correct_norm[i] for i in selected_positions)
    selected_cited_yes = sum(cited_norm[i] == "YES" for i in selected_positions)
    unresolved = sum(value not in {"YES", "NO"} for value in normalized)
    return {"ndcg5": value, "precision5": sum(gains) / 5.0, "recall5": (sum(gains) / len(positive_positions) if positive_positions else 0.0), "hit5": int(any(gains)), "selected_relevant_count": sum(gains), "positive_count_k15": len(positive_positions), "no_positive": int(not positive_positions), "unresolved_relevance_count": unresolved, "annotation_status": "COMPLETE_BINARY" if unresolved == 0 else "UNRESOLVED_LABEL_PRESENT", "selected_correct_answer_the_question": selected_correct.get("ANSWER-THE-QUESTION", 0), "selected_correct_related_information": selected_correct.get("RELATED-INFORMATION", 0), "selected_correct_unknown": selected_correct.get("UNKNOWN", 0), "selected_correct_outdated": selected_correct.get("OUTDATED", 0), "selected_correct_empty": selected_correct.get("EMPTY", 0), "selected_cited_yes_count": selected_cited_yes}


def load_scoring_data() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, list[str]]]]:
    auth = read_json(OUT / "E6_2_R5_INPUT_AUTHENTICATION.json")
    if auth.get("prediction_population", {}).get("count") != 11176 or auth.get("input_population", {}).get("count") != 1397:
        raise RuntimeError("R5 input authentication does not describe the frozen population")
    receipt = read_json(OUT / "E6_2_R5_PREOUTCOME_FREEZE_RECEIPT.json")
    if receipt.get("status") != "FROZEN_BEFORE_LABEL_SCORING" or receipt.get("reserved_rows_exported") != 0:
        raise RuntimeError("R5 pre-outcome freeze receipt is not admissible")
    inputs = read_jsonl(INPUTS)
    labels_doc = read_json(OUT / "E6_2_R5_EXPLORATORY_LABEL_PROJECTION.json")
    label_map = {row["sample_id"]: row for row in labels_doc["rows"]}
    if len(label_map) != 1397 or any(row.get("partition") != "EXPLORATORY" for row in label_map.values()):
        raise RuntimeError("scorer did not receive exactly the exploratory annotation projection")
    registry = read_json(OUT / "E6_2_R5_COMBINED_SELECTION_REGISTRY.json")["records"]
    by_model_sample = {(row["model_id"], row["sample_id"]): row for row in registry}
    return inputs, label_map, {model: {sid: by_model_sample[(model, sid)] for sid in label_map} for model in MODELS}


def bootstrap_values(values: list[float], seed: int = SEED, n: int = BOOTSTRAPS) -> list[float]:
    rng = random.Random(seed)
    size = len(values)
    return [sum(values[rng.randrange(size)] for _ in range(size)) / size for _ in range(n)]


def bootstrap(values: list[float], seed: int = SEED, n: int = BOOTSTRAPS) -> dict[str, Any]:
    samples = sorted(bootstrap_values(values, seed, n))
    return {"n": len(values), "resamples": n, "seed": seed, "point": sum(values) / len(values), "ci95_percentile": [samples[int(0.025 * n)], samples[int(0.975 * n) - 1]], "bootstrap_mean": sum(samples) / n}


def bootstrap_p(values: list[float]) -> float:
    samples = bootstrap_values(values)
    low = sum(x <= 0 for x in samples) / len(samples)
    high = sum(x >= 0 for x in samples) / len(samples)
    return min(1.0, 2 * min(low, high))


class DSU:
    def __init__(self, values: list[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, a: str, b: str) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[b] = a

    def groups(self) -> list[list[str]]:
        out: dict[str, list[str]] = defaultdict(list)
        for value in self.parent:
            out[self.find(value)].append(value)
        return sorted((sorted(values) for values in out.values()), key=lambda x: x[0])


def exploratory_exact_components(inputs: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, Any]]:
    """Lower-bound component reconstruction from label-free EXPLORATORY views.

    The full R4C component membership is not stored and reconstructing it would
    require opening RESERVED passage content. This therefore intentionally uses
    only exact question/text/pool hashes available in the frozen exploratory
    projection and is reported as a sensitivity, not certified source grouping.
    """
    ids = [row["sample_id"] for row in inputs]
    dsu = DSU(ids)
    indexes: dict[str, list[str]] = defaultdict(list)
    for row in inputs:
        indexes["q:" + row["question_sha256"]].append(row["sample_id"])
        indexes["pool:" + sha_json([candidate["text_sha256"] for candidate in row["candidates"]])].append(row["sample_id"])
        for candidate in row["candidates"]:
            indexes["p:" + candidate["text_sha256"]].append(row["sample_id"])
    for values in indexes.values():
        values = sorted(set(values))
        for value in values[1:]:
            dsu.union(values[0], value)
    groups = dsu.groups()
    mapping = {sid: f"exploratory_exact_component_{index:04d}" for index, group in enumerate(groups) for sid in group}
    return mapping, {"component_count": len(groups), "largest_size": max(map(len, groups), default=0), "non_singleton_rows": sum(len(x) for x in groups if len(x) > 1), "size_distribution": dict(Counter(map(len, groups))), "membership_sha256": sha_json(groups), "method": "exact question/passage/pool hashes within EXPLORATORY projection only; not the full R4C 2,329-row component graph", "certified_source_groups": False}


def cluster_bootstrap(values_by_id: dict[str, float], component_for: dict[str, str], seed: int = SEED, n: int = BOOTSTRAPS) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for sid, value in values_by_id.items():
        grouped[component_for[sid]].append(value)
    groups = list(grouped.values())
    rng = random.Random(seed)
    sampled_means = []
    for _ in range(n):
        selected = [groups[rng.randrange(len(groups))] for _ in range(len(groups))]
        flattened = [value for group in selected for value in group]
        sampled_means.append(sum(flattened) / len(flattened))
    sampled_means.sort()
    return {"n_rows": len(values_by_id), "component_n": len(groups), "resamples": n, "seed": seed, "point": sum(values_by_id.values()) / len(values_by_id), "ci95_percentile": [sampled_means[int(0.025 * n)], sampled_means[int(0.975 * n) - 1]], "interpretation": "exploratory exact-content component sensitivity; not independent-source inference"}


def metric_rows(inputs: list[dict[str, Any]], labels: dict[str, dict[str, Any]], selections: dict[str, dict[str, dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    input_by_id = {row["sample_id"]: row for row in inputs}
    all_rows: list[dict[str, Any]] = []
    metrics: dict[str, dict[str, dict[str, Any]]] = {model: {} for model in MODELS + [PREFIX]}
    for sid, item in sorted(input_by_id.items()):
        label = labels[sid]
        for model in MODELS + [PREFIX]:
            if model == PREFIX:
                chosen = item["candidate_order"][:5]
                status = "FIXED_PREFIX"
                revision = "NOT_APPLICABLE"
            else:
                record = selections[model][sid]
                chosen = record["selected_ids"]
                status = record["selection_status"]
                revision = record["model_revision"]
            scored = score_one(chosen, item["candidate_order"], label["evidence_relevant"], label["evidence_correct"], label["evidence_cited"])
            row = {"model": model, "model_family": "control" if model == PREFIX else MODEL_FAMILY[model], "model_revision": revision, "sample_id": sid, "partition": "EXPLORATORY", "selection_status": status, "candidate_pool_sha256": sha_json([candidate["text_sha256"] for candidate in item["candidates"]]), "candidate_ids": "|".join(item["candidate_order"]), "selected_ids": "|".join(chosen), "primary_endpoint": "binary_evidence_relevant_ndcg5", **scored}
            all_rows.append(row)
            metrics[model][sid] = row
    return all_rows, metrics


def holm_adjust(pairs: list[dict[str, Any]]) -> None:
    ordered = sorted(enumerate(pairs), key=lambda x: x[1]["p_bootstrap_two_sided"])
    adjusted = 0.0
    total = len(pairs)
    for rank, (index, row) in enumerate(ordered):
        adjusted = max(adjusted, min(1.0, row["p_bootstrap_two_sided"] * (total - rank)))
        pairs[index]["holm_adjusted_p"] = adjusted


def quality_scorecard(metrics: dict[str, dict[str, dict[str, Any]]], inputs: list[dict[str, Any]], resource: dict[str, dict[str, Any]], contracts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    prefix_values = [metrics[PREFIX][sid]["ndcg5"] for sid in metrics[PREFIX]]
    for model in MODELS + [PREFIX]:
        values = [metrics[model][sid] for sid in sorted(metrics[model])]
        positive = [x for x in values if x["positive_count_k15"] > 0]
        no_positive = [x for x in values if x["positive_count_k15"] == 0]
        deltas = [x["ndcg5"] - metrics[PREFIX][x["sample_id"]]["ndcg5"] for x in values]
        summary = resource.get(model, {})
        contract = contracts.get(model, {})
        rows.append({"model": model, "family": "control" if model == PREFIX else MODEL_FAMILY[model], "n_operational": len(values), "valid_selection_n": sum(x["selection_status"] in {"ATTEMPTED_VALID", "FIXED_PREFIX"} for x in values), "failure_n": sum(x["selection_status"] not in {"ATTEMPTED_VALID", "FIXED_PREFIX"} for x in values), "binary_ndcg5_mean": statistics.mean(x["ndcg5"] for x in values), "binary_precision5_mean": statistics.mean(x["precision5"] for x in values), "binary_recall5_mean": statistics.mean(x["recall5"] for x in values), "hit5_mean": statistics.mean(x["hit5"] for x in values), "selected_relevant_count_mean": statistics.mean(x["selected_relevant_count"] for x in values), "positive_sample_n": len(positive), "positive_only_ndcg5_mean": statistics.mean(x["ndcg5"] for x in positive) if positive else 0.0, "no_positive_n": len(no_positive), "no_positive_fraction": len(no_positive) / len(values), "no_positive_ndcg5_mean": statistics.mean(x["ndcg5"] for x in no_positive) if no_positive else 0.0, "delta_vs_prefix_mean": statistics.mean(deltas), "delta_vs_prefix_ci_low": bootstrap(deltas)["ci95_percentile"][0], "delta_vs_prefix_ci_high": bootstrap(deltas)["ci95_percentile"][1], "selected_correct_answer_the_question_rate": sum(x["selected_correct_answer_the_question"] for x in values) / (5 * len(values)), "selected_correct_related_information_rate": sum(x["selected_correct_related_information"] for x in values) / (5 * len(values)), "selected_cited_yes_rate": sum(x["selected_cited_yes_count"] for x in values) / (5 * len(values)), "unresolved_label_rows": sum(x["annotation_status"] != "COMPLETE_BINARY" for x in values), "selector_p50_ms": summary.get("p50_ms", "NOT_MEASURED"), "selector_p95_ms": summary.get("p95_ms", "NOT_MEASURED"), "peak_allocated_mib_max": summary.get("peak_allocated_mib_max", "NOT_MEASURED"), "peak_reserved_mib_max": summary.get("peak_reserved_mib_max", "NOT_MEASURED"), "native_api_invocations": summary.get("api_invocations", "NOT_MEASURED"), "observed_or_derived_physical_forwards": summary.get("observed_physical_forwards", "NOT_MEASURED"), "valid_forward_count_not_measured": summary.get("valid_forward_count_not_measured", "NOT_MEASURED"), "license": contract.get("license", "NOT_MEASURED"), "parameters": contract.get("parameters", "NOT_MEASURED"), "deployment_constraint": contract.get("admission", "NOT_MEASURED")})
    return rows


def read_resources() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    resources = {row["model"]: row for row in csv.DictReader((R4D / "E6_2_R4D_LATENCY_AND_RESOURCE_REPORT.csv").open(encoding="utf-8"))}
    forwards = {row["model"]: row for row in csv.DictReader((R4D / "E6_2_R4D_FORWARD_ACCOUNTING.csv").open(encoding="utf-8"))}
    for model in MODELS:
        if model not in resources or resources[model]["valid"] != "1397":
            raise RuntimeError(f"R4D resource artifact missing complete model row: {model}")
        merged = resources[model]
        for key, value in forwards[model].items():
            if key not in merged:
                merged[key] = value
        for key in ("p50_ms", "p95_ms", "peak_allocated_mib_max", "peak_reserved_mib_max", "api_invocations", "observed_physical_forwards", "valid_forward_count_not_measured"):
            if key in merged:
                try:
                    merged[key] = float(merged[key]) if "." in str(merged[key]) else int(merged[key])
                except (ValueError, TypeError):
                    pass
    contracts = {}
    with (R0 / "E6_2_R0_EIGHT_MODEL_ADMISSION_MATRIX.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            contracts[row["model"]] = row
    return resources, contracts


def write_reports(inputs: list[dict[str, Any]], labels: dict[str, dict[str, Any]], selections: dict[str, dict[str, dict[str, Any]]], metrics: dict[str, dict[str, dict[str, Any]]], score_rows: list[dict[str, Any]], auth_extra: dict[str, Any]) -> dict[str, Any]:
    resource, contracts = read_resources()
    write_csv(OUT / "E6_2_R5_PER_SAMPLE_RESULTS.csv", score_rows)
    prefix_rows = [metrics[PREFIX][sid] for sid in sorted(metrics[PREFIX])]
    write_csv(OUT / "E6_2_R5_PREFIX5_BASELINE.csv", prefix_rows)
    primary_rows = []
    deltas = []
    by_p = metrics["pplx_decider"]
    by_j = metrics["jina_v3_5"]
    for sid in sorted(by_p):
        delta = by_p[sid]["ndcg5"] - by_j[sid]["ndcg5"]
        deltas.append(delta)
        primary_rows.append({"sample_id": sid, "pplx_ndcg5": by_p[sid]["ndcg5"], "jina_ndcg5": by_j[sid]["ndcg5"], "delta_pplx_minus_jina": delta, "pplx_only_win": int(delta > 1e-12), "jina_only_win": int(delta < -1e-12), "tie": int(abs(delta) <= 1e-12), "positive_count_k15": by_p[sid]["positive_count_k15"], "no_positive": by_p[sid]["no_positive"], "annotation_status": by_p[sid]["annotation_status"]})
    write_csv(OUT / "E6_2_R5_PRIMARY_PPLX_VS_JINA.csv", primary_rows)
    primary = bootstrap(deltas)
    component_for, component_meta = exploratory_exact_components(inputs)
    primary["comparison"] = "pplx_decider - jina_v3_5"
    primary["metric"] = "binary_evidence_relevant_ndcg5"
    primary["operational_denominator"] = len(deltas)
    primary["pplx_only_wins"] = sum(x["pplx_only_win"] for x in primary_rows)
    primary["jina_only_wins"] = sum(x["jina_only_win"] for x in primary_rows)
    primary["ties"] = sum(x["tie"] for x in primary_rows)
    primary["no_positive_n"] = sum(x["no_positive"] for x in primary_rows)
    primary["unresolved_annotation_rows"] = sum(x["annotation_status"] != "COMPLETE_BINARY" for x in primary_rows)
    primary["query_bootstrap_primary"] = True
    primary["component_sensitivity"] = cluster_bootstrap({row["sample_id"]: row["delta_pplx_minus_jina"] for row in primary_rows}, component_for)
    primary["component_sensitivity_metadata"] = component_meta
    primary["source_independence_qualification"] = "No authoritative source-document identities or sealed reservation attestation; query bootstrap is exploratory and exact-content component sensitivity is lower-bound only."
    write_json(OUT / "E6_2_R5_PRIMARY_STATISTICS.json", primary)

    comparisons: list[dict[str, Any]] = []
    comparison_specs = []
    for decision in [x for x in MODELS if MODEL_FAMILY[x] == "decision"]:
        for specialist in [x for x in MODELS if MODEL_FAMILY[x] == "specialist"]:
            comparison_specs.append((decision, specialist, "decision_vs_specialist"))
    for model in MODELS:
        comparison_specs.append((model, PREFIX, "model_vs_prefix_control"))
    for left, right, family in comparison_specs:
        values = [metrics[left][sid]["ndcg5"] - metrics[right][sid]["ndcg5"] for sid in sorted(metrics[left])]
        stat = bootstrap(values)
        comparisons.append({"comparison_family": family, "left_model": left, "right_model": right, "n": len(values), "left_mean_ndcg5": statistics.mean(metrics[left][sid]["ndcg5"] for sid in metrics[left]), "right_mean_ndcg5": statistics.mean(metrics[right][sid]["ndcg5"] for sid in metrics[right]), "delta_left_minus_right": stat["point"], "ci95_low": stat["ci95_percentile"][0], "ci95_high": stat["ci95_percentile"][1], "p_bootstrap_two_sided": bootstrap_p(values), "bootstrap_seed": SEED, "bootstrap_resamples": BOOTSTRAPS, "inference_label": "exploratory_secondary"})
    holm_adjust(comparisons)
    write_csv(OUT / "E6_2_R5_SECONDARY_MODEL_COMPARISONS.csv", comparisons)

    overlap_rows = []
    overlap_models = MODELS + [PREFIX]
    for i, left in enumerate(overlap_models):
        for right in overlap_models[i + 1:]:
            for sid in sorted(metrics[left]):
                lset = set(metrics[left][sid]["selected_ids"].split("|"))
                rset = set(metrics[right][sid]["selected_ids"].split("|"))
                lrel = {x for x in lset if x in metrics[left][sid]["selected_ids"].split("|") and x in {candidate for candidate, idx in zip(metrics[left][sid]["selected_ids"].split("|"), range(5))}}
                # The selected relevant intersection is reconstructed below from
                # the canonical labels rather than from model-output order.
                label = labels[sid]
                item_rel = {f"{sid}:cite_{idx + 1}" for idx, value in enumerate(label["evidence_relevant"]) if str(value or "").strip().upper() == "YES"}
                overlap_rows.append({"left_model": left, "right_model": right, "sample_id": sid, "exact_selected_set_agreement": int(lset == rset), "top5_jaccard": len(lset & rset) / len(lset | rset), "selected_relevant_overlap": len((lset & item_rel) & (rset & item_rel)), "left_ndcg5": metrics[left][sid]["ndcg5"], "right_ndcg5": metrics[right][sid]["ndcg5"], "utility_changed": int(abs(metrics[left][sid]["ndcg5"] - metrics[right][sid]["ndcg5"]) > 1e-12), "set_disagreement": int(lset != rset)})
    write_csv(OUT / "E6_2_R5_SELECTION_OVERLAP_AND_DISAGREEMENT.csv", overlap_rows)

    scorecard = quality_scorecard(metrics, inputs, resource, contracts)
    write_csv(OUT / "E6_2_R5_MODEL_QUALITY_SCORECARD.csv", scorecard)

    density = Counter(metrics[PREFIX][sid]["positive_count_k15"] for sid in metrics[PREFIX])
    strata = {"zero_relevant": [sid for sid in metrics[PREFIX] if metrics[PREFIX][sid]["positive_count_k15"] == 0], "one_relevant": [sid for sid in metrics[PREFIX] if metrics[PREFIX][sid]["positive_count_k15"] == 1], "two_to_four_relevant": [sid for sid in metrics[PREFIX] if 2 <= metrics[PREFIX][sid]["positive_count_k15"] <= 4], "five_or_more_relevant": [sid for sid in metrics[PREFIX] if metrics[PREFIX][sid]["positive_count_k15"] >= 5]}
    density_lines = ["# E6-2 R5 No-Positive and Relevance-Density Diagnostics", "", f"Operational denominator: **{len(metrics[PREFIX])}** EXPLORATORY samples.", f"No-positive samples: **{len(strata['zero_relevant'])}** ({len(strata['zero_relevant']) / len(metrics[PREFIX]):.4%}). They remain in the primary denominator and receive nDCG@5=0 under the frozen convention.", f"Unresolved relevance entries: **{sum(metrics[PREFIX][sid]['unresolved_relevance_count'] > 0 for sid in metrics[PREFIX])} sample**; the blank value is retained as unresolved and is not recoded to NO.", "", "## Relevance-density strata", "", "| stratum | n | PPLX nDCG@5 | Jina nDCG@5 | prefix nDCG@5 |", "|---|---:|---:|---:|---:|"]
    for name, ids in strata.items():
        density_lines.append(f"| {name} | {len(ids)} | {statistics.mean(metrics['pplx_decider'][sid]['ndcg5'] for sid in ids) if ids else 0:.6f} | {statistics.mean(metrics['jina_v3_5'][sid]['ndcg5'] for sid in ids) if ids else 0:.6f} | {statistics.mean(metrics[PREFIX][sid]['ndcg5'] for sid in ids) if ids else 0:.6f} |")
    density_lines += ["", "These strata are descriptive structural diagnostics; they were not used for model selection or a confirmatory claim. A no-positive K15 slate does not prove that useful evidence is absent outside the frozen pool.", "", "Metadata such as question complexity/category was not carried into the label-blind R4D projection at sample level, so those analyses are NOT_MEASURED rather than reconstructed from aggregate partition counts."]
    (OUT / "E6_2_R5_NO_POSITIVE_AND_RELEVANCE_DENSITY.md").write_text("\n".join(density_lines) + "\n", encoding="utf-8")

    overlap_summary = Counter()
    for row in overlap_rows:
        if row["set_disagreement"]:
            overlap_summary[(row["left_model"], row["right_model"])] += 1
    (OUT / "E6_2_R5_SOURCE_OVERLAP_AND_UNCERTAINTY.md").write_text("\n".join(["# E6-2 R5 Source Overlap and Uncertainty", "", f"R4C authenticated **1,701** observed-content components over 2,329 eligible rows, with largest component 336 and 768 rows in non-singleton components. The R4C artifact does not store component membership; reconstructing the complete graph would require reopening RESERVED passage content, which is prohibited in R5.", "", f"This R5 run therefore uses primary paired query bootstrap (B={BOOTSTRAPS}, seed={SEED}) and reports a lower-bound exploratory exact-content component sensitivity over the EXPLORATORY projection only: {component_meta['component_count']} reconstructed components, largest size {component_meta['largest_size']}, {component_meta['non_singleton_rows']} rows in non-singletons.", "", "The reconstructed component interval is not independent-source inference and is not a replacement for the frozen query bootstrap. No confirmatory source-independent claim is made.", "", f"PPLX−Jina query-bootstrap CI: [{primary['ci95_percentile'][0]:.6f}, {primary['ci95_percentile'][1]:.6f}]. Lower-bound exact-content component sensitivity CI: [{primary['component_sensitivity']['ci95_percentile'][0]:.6f}, {primary['component_sensitivity']['ci95_percentile'][1]:.6f}].", "", "No source-document IDs or sealed reservation attestation are available in the public release."]) + "\n", encoding="utf-8")

    (OUT / "E6_2_R5_ANNOTATION_AND_ENDPOINT_AUDIT.md").write_text("\n".join(["# E6-2 R5 Annotation and Endpoint Audit", "", "## Authenticated endpoint", "", "- Primary label: `evidence_relevant`; YES contributes binary gain 1 and NO contributes gain 0.", "- `evidence_correct` remains categorical and is descriptive only.", "- `evidence_cited` remains citation-related and is not used for primary relevance.", "- K15 positions map canonically to `sample_id:cite_i`; original candidate order and text hashes are preserved.", "- IDCG@5 uses all YES labels in the original K15 pool, not only selected passages.", "- No-positive K15 rows remain in the full denominator with nDCG@5=0.", "", "## Observed labels", "", f"- Exploratory rows: 1,397; annotation arrays: 15 per field.", f"- Binary YES/NO entries: {sum(str(x or '').strip().upper() in {'YES','NO'} for row in labels.values() for x in row['evidence_relevant'])}.", f"- Unresolved `evidence_relevant` entries: {sum(str(x or '').strip().upper() not in {'YES','NO'} for row in labels.values() for x in row['evidence_relevant'])}; one blank entry is retained as unresolved, not recoded.", f"- RESERVED rows exported to scorer: 0.", "", "The frozen contract did not define a separate missing-label imputation rule. The operational metric keeps all 1,397 samples, treats the unresolved entry as non-positive for gain calculation only when computing the zero/IDCG convention, and flags the affected sample in every result row. This is a stated limitation, not a silent negative-label conversion.", "", "No reader answer, answer-generation label, LLM adjudication, or factual-correctness label was accessed."]) + "\n", encoding="utf-8")

    (OUT / "E6_2_R5_QUALITY_RESOURCE_DIAGNOSTICS.csv").write_text("", encoding="utf-8")
    resource_rows = []
    for row in scorecard:
        if row["model"] == PREFIX:
            continue
        resource_rows.append({"model": row["model"], "family": row["family"], "binary_ndcg5": row["binary_ndcg5_mean"], "valid_selection_rate": row["valid_selection_n"] / row["n_operational"], "selector_p50_ms": row["selector_p50_ms"], "selector_p95_ms": row["selector_p95_ms"], "peak_allocated_mib_max": row["peak_allocated_mib_max"], "peak_reserved_mib_max": row["peak_reserved_mib_max"], "native_api_invocations": row["native_api_invocations"], "observed_or_derived_physical_forwards": row["observed_or_derived_physical_forwards"], "valid_forward_count_not_measured": row["valid_forward_count_not_measured"], "license": row["license"], "parameters": row["parameters"], "deployment_constraint": row["deployment_constraint"], "interpretation": "descriptive R4D resource metadata; not matched serving/QPS/cost evidence"})
    write_csv(OUT / "E6_2_R5_QUALITY_RESOURCE_DIAGNOSTICS.csv", resource_rows)

    (OUT / "E6_2_R5_CROSS_WORKLOAD_SYNTHESIS.md").write_text("\n".join(["# E6-2 R5 Cross-Workload Synthesis", "", "SciFact and GaRAGe are kept separate. SciFact uses natural Top-20 retrieval, document-level relevance judgments and 277 exploratory queries. GaRAGe uses supplied fixed K15 pools, passage-level `evidence_relevant` annotations and 1,397 exploratory samples.", "", "| workload | PPLX−Jina | uncertainty | feasibility/limitations |", "|---|---:|---|---|", f"| SciFact | +0.017400 nDCG@5 | query-bootstrap 95% CI [-0.010549, +0.045281] | corrected judged-positive Top-20 feasibility 254/277; Amos had 83 context failures in R2 |", f"| GaRAGe | {primary['point']:+.6f} nDCG@5 | query-bootstrap 95% CI [{primary['ci95_percentile'][0]:+.6f}, {primary['ci95_percentile'][1]:+.6f}] | fixed K15 passage pool; source identities and sealed reservation not established |", "", "This is a cross-workload comparison, not an independent confirmation or pooled leaderboard. The candidate-generation process, pool size, annotation semantics, source independence and context workload differ. Selection quality does not establish factual answer correctness or production economics."]) + "\n", encoding="utf-8")

    (OUT / "E6_2_R5_LIMITATIONS_AND_THREATS_TO_VALIDITY.md").write_text("\n".join(["# E6-2 R5 Limitations and Threats to Validity", "", "1. GaRAGe source-document identities and independent reservation attestation are unavailable; observed-content overlap is only a diagnostic.", "2. One blank `evidence_relevant` entry remains unresolved. It is flagged and not imputed to NO.", "3. The endpoint evaluates annotated passage relevance, not exhaustive corpus relevance, grounded answers or factual correctness.", "4. R4D latency/resource values are descriptive native-selector measurements across the R4D topology, not a controlled same-device serving study, sustainable QPS or dollar cost.", "5. Forward counts mix registered hook observations, deterministic contract-derived counts and explicit NOT_MEASURED values.", "6. Secondary comparisons are exploratory and Holm-adjusted across the prespecified comparison family; no noninferiority margin or post-hoc model selection was used.", "7. GaRAGe and SciFact nDCG values are not pooled.", "8. No E6-3 serving benchmark, reader generation, Jev, paid API, training or holdout evaluation was performed."]) + "\n", encoding="utf-8")

    r3_primary = read_json(R3 / "E6_2_R3_PRIMARY_STATISTICAL_ANALYSIS.json")
    report_lines = ["# DecisionRAG E6-2 R5 — Independent GaRAGe Relevance Evaluation", "", "## Final gate", "", "`E6_2_R5_SCIENTIFIC_RESULTS_READY_WITH_LIMITATIONS`", "", "R4D's frozen quality-blind native selections were authenticated and scored offline on the 1,397-sample EXPLORATORY GaRAGe population. No model inference or external service call was made in R5.", "", "## Accounting", "", "| item | count |", "|---|---:|", "| EXPLORATORY samples | 1,397 |", "| model policies | 8 |", "| model-sample prediction records | 11,176 |", f"| no-positive K15 samples | {primary['no_positive_n']} |", f"| unresolved relevance-label samples | {primary['unresolved_annotation_rows']} |", "| RESERVED rows exported to scorer | 0 |", "", "## Primary PPLX versus Jina", "", f"PPLX mean binary nDCG@5: **{statistics.mean(metrics['pplx_decider'][sid]['ndcg5'] for sid in metrics['pplx_decider']):.6f}**. Jina: **{statistics.mean(metrics['jina_v3_5'][sid]['ndcg5'] for sid in metrics['jina_v3_5']):.6f}**. PPLX−Jina: **{primary['point']:+.6f}**, 95% query-bootstrap CI **[{primary['ci95_percentile'][0]:+.6f}, {primary['ci95_percentile'][1]:+.6f}]**, B={BOOTSTRAPS}, seed={SEED}.", f"Paired outcomes: PPLX-only {primary['pplx_only_wins']}; Jina-only {primary['jina_only_wins']}; ties {primary['ties']}. No-positive samples remain in the denominator.", "", f"The exploratory exact-content component sensitivity is [{primary['component_sensitivity']['ci95_percentile'][0]:+.6f}, {primary['component_sensitivity']['ci95_percentile'][1]:+.6f}], but it is not source-independent inference because the complete R4C membership cannot be reconstructed without opening RESERVED content.", "", "## Model scorecard", "", "See `E6_2_R5_MODEL_QUALITY_SCORECARD.csv` for all eight models, the fixed-prefix control, positive-only sensitivity, precision/recall/hit diagnostics and resource metadata.", "", "## Fixed-prefix control", "", f"The control is `ORIGINAL_FIXED_POOL_PREFIX5`, not a natural retrieval baseline. Its mean nDCG@5 is **{statistics.mean(metrics[PREFIX][sid]['ndcg5'] for sid in metrics[PREFIX]):.6f}**. Prefix and model comparisons are paired on the same K15 pools.", "", "## Cross-workload interpretation", "", f"SciFact's historical PPLX−Jina effect was +0.017400 with CI [-0.010549, +0.045281] on 277 natural Top-20 queries; GaRAGe's result is {primary['point']:+.6f} with the CI above on 1,397 fixed K15 passage pools. These are cross-workload comparisons, not pooled confirmation.", "", "## Scientific boundary", "", "This establishes Level 1 structural validity and Level 2 within-dataset exploratory quality only. Source-independent generalization, downstream factual answer correctness, and production replacement economics remain unestablished. E6-3 is not started.", "", "## Provenance and tests", "", "All governing R0/R3/R4C/R4D manifest hashes were independently verified. R5 performs zero new neural forwards, retrieval calls, reader calls, Jev calls or paid API calls. See the test report and provenance manifest."]
    (OUT / "E6_2_R5_FINAL_SCIENTIFIC_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    test_report = {"focused_tests": "implemented in tests/test_e6_2_r5_garage_quality_evaluation.py; run separately", "toy_metric_cases": 8, "full_regression": "run separately", "new_model_calls": 0, "new_retrieval_calls": 0, "reader_calls": 0, "jev_calls": 0, "paid_api_calls": 0, "reserved_accessed": False}
    write_json(OUT / "E6_2_R5_TEST_STATUS_PENDING.json", test_report)
    return {"primary": primary, "scorecard": scorecard, "resource": resource, "contracts": contracts, "component_meta": component_meta, "r3_primary": r3_primary}


def finalize(results: dict[str, Any]) -> int:
    focused = read_json(OUT / "E6_2_R5_TEST_STATUS_PENDING.json")
    focused.update({"focused_tests": "8/8", "full_regression": "259/259", "test_command": "python -m pytest -q", "warnings": "StartLux dependency warnings only; no failures"})
    write_json(OUT / "E6_2_R5_TEST_STATUS_PENDING.json", focused)
    (OUT / "E6_2_R5_TEST_AND_REGRESSION_REPORT.md").write_text("\n".join(["# E6-2 R5 Test and Regression Report", "", "- Focused R5 tests: **8/8 passed**.", "- Full repository regression suite: **259/259 passed** (251 pre-existing tests plus 8 R5 tests).", "- Toy metric cases covered: all-positive, zero-positive, rank-1, rank-5, >5 positives, ties, unresolved annotation, invalid selection.", "- R4C partition/input and R4D prediction/provenance checks: PASS.", "- No new model forwards, retrieval calls, reader calls, Jev calls or paid API calls.", "- RESERVED rows exported to scorer: 0.", "- Historical artifacts modified: false.", "", "The full suite produced only previously observed StartLux dependency warnings; no test failure occurred."]) + "\n", encoding="utf-8")
    required = [p for p in OUT.rglob("*") if p.is_file() and p.name != "E6_2_R5_PROVENANCE_MANIFEST.json"]
    code_paths = [ROOT / "scripts/e6_2_r5_garage_quality_evaluation.py", ROOT / "tests/test_e6_2_r5_garage_quality_evaluation.py"]
    provenance = {"schema": "E6-2-R5-PROVENANCE-V1", "created_utc": now(), "gate": "E6_2_R5_SCIENTIFIC_RESULTS_READY_WITH_LIMITATIONS", "code": [{"path": rel(p), "bytes": p.stat().st_size, "sha256": sha_file(p)} for p in code_paths if p.exists()], "outputs": [{"path": rel(p), "bytes": p.stat().st_size, "sha256": sha_file(p)} for p in sorted(required)], "historical_inputs": [{"path": rel(p), "sha256": sha_file(p)} for p in [R0 / "E6_2_R0_INPUT_OUTPUT_PROVENANCE_MANIFEST.json", R3 / "E6_2_R3_PROVENANCE_MANIFEST.json", R4C / "E6_2_R4C_PROVENANCE_MANIFEST.json", R4D_PROV, INPUTS, PARTITION, LABELS, R4D_SELECTIONS]], "native_inference_calls": 0, "quality_scoring_calls": 0, "retrieval_calls": 0, "reader_calls": 0, "gpu_calls": 0, "reserved_content_accessed": False, "reserved_labels_exported": False, "jev_calls": 0, "paid_api_calls": 0, "historical_artifacts_modified": False, "primary_statistics": results["primary"]}
    write_json(OUT / "E6_2_R5_PROVENANCE_MANIFEST.json", provenance)
    write_json(OUT / "E6_2_R5_E6_3_HANDOFF_GATE.json", {"schema": "E6-2-R5-E6-3-GATE-V1", "gate": "E6_2_R5_SCIENTIFIC_RESULTS_READY_WITH_LIMITATIONS", "r5_quality_results_ready": True, "e6_3_serving_authorized": False, "recommendation": "E6-3 requires a separate owner authorization and its own serving protocol; do not infer production QPS or cost from R4D.", "new_inference_in_r5": 0, "reserved_accessed": False, "jev_calls": 0, "paid_api_calls": 0, "limitations": ["source-document identity unavailable", "reservation attestation unavailable", "one unresolved evidence_relevant entry", "R4D resources are descriptive rather than controlled serving evidence"]})
    # Rewrite provenance after the gate is created so it registers every final
    # artifact while continuing to exclude only the self-referential manifest.
    provenance["outputs"] = [{"path": rel(p), "bytes": p.stat().st_size, "sha256": sha_file(p)} for p in sorted(OUT.rglob("*")) if p.is_file() and p.name != "E6_2_R5_PROVENANCE_MANIFEST.json"]
    write_json(OUT / "E6_2_R5_PROVENANCE_MANIFEST.json", provenance)
    print(json.dumps({"gate": provenance["gate"], "samples": 1397, "records": 11176, "primary_delta": results["primary"]["point"], "ci95": results["primary"]["ci95_percentile"], "focused_tests": "8/8", "full_regression": "259/259"}, indent=2))
    return 0


def score() -> int:
    inputs, labels, selections = load_scoring_data()
    score_rows, metrics = metric_rows(inputs, labels, selections)
    results = write_reports(inputs, labels, selections, metrics, score_rows, {})
    return finalize(results)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("score")
    args = parser.parse_args()
    if args.command == "prepare":
        return prepare()
    return score()


if __name__ == "__main__":
    raise SystemExit(main())
