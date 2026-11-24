"""CPU-only E2-N adapter registry and fail-closed scoring contracts.

The registry describes interfaces; it does not load a checkpoint or make a
model request.  It is intentionally usable by future GPU adapters without
coupling the contract layer to torch, transformers, network clients, or a
particular serving runtime.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping


CONTRACT_VERSION = "e2-n-n0-contract-v1"


@dataclass(frozen=True)
class AdapterSpec:
    adapter_id: str
    model_id: str
    status: str
    score_contracts: tuple[str, ...]
    native_interface: str
    revision: str
    checkpoint_path: str


ADAPTER_REGISTRY: tuple[AdapterSpec, ...] = (
    AdapterSpec("clef_flash_native", "Cloudflare/clef-flash", "VERIFIED_LOCAL", ("INDEPENDENT_RELEVANCE", "JOINT_BINARY_RELEVANCE", "LOCAL_CATEGORICAL"), "joint_schema_model.py", "8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3", ""),
    AdapterSpec("laya_typed", "convaiinnovations/laya", "VERIFIED_LOCAL", ("INDEPENDENT_RELEVANCE", "LOCAL_CATEGORICAL"), "typed decision router", "7b928d828b7b0e022f929d9bd2e44165aa270148", ""),
    AdapterSpec("bge_pair", "BAAI/bge-reranker-v2-m3", "VERIFIED_LOCAL", ("INDEPENDENT_RELEVANCE",), "FlagReranker query/passage pair", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e", ""),
    AdapterSpec("qwen3_vl_pair", "Qwen/Qwen3-VL-Reranker-2B", "VERIFIED_LOCAL", ("INDEPENDENT_RELEVANCE",), "official Qwen3-VL reranker script", "4bd860ac4f15ad1897a214615cccc700f8f71818", ""),
)


def registry_snapshot() -> dict[str, Any]:
    return {"contract_version": CONTRACT_VERSION, "inference_enabled": False, "adapters": [asdict(x) for x in ADAPTER_REGISTRY]}


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_slate(question: str, candidate_ids: Iterable[str], expected_k: int = 20) -> None:
    if not str(question).strip():
        raise ValueError("empty question")
    ids = list(candidate_ids)
    if len(ids) != expected_k:
        raise ValueError(f"expected {expected_k} candidates, got {len(ids)}")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate candidate IDs")
    if any(not str(x).strip() for x in ids):
        raise ValueError("empty candidate ID")


def validate_scores(candidate_ids: Iterable[str], scores: Mapping[str, float], *, score_direction: str = "higher_is_more_relevant") -> list[str]:
    ids = list(candidate_ids)
    if set(scores) != set(ids) or len(scores) != len(ids):
        raise ValueError("score IDs do not exactly match candidate IDs")
    if score_direction not in {"higher_is_more_relevant", "lower_is_more_relevant"}:
        raise ValueError("unknown score direction")
    for doc_id in ids:
        value = float(scores[doc_id])
        if not math.isfinite(value):
            raise ValueError(f"nonfinite score for {doc_id}")
    sign = -1.0 if score_direction == "higher_is_more_relevant" else 1.0
    return sorted(ids, key=lambda x: (sign * float(scores[x]), ids.index(x), x))


def select_top5(candidate_ids: Iterable[str], scores: Mapping[str, float], *, score_direction: str = "higher_is_more_relevant") -> list[str]:
    ids = list(candidate_ids)
    validate_slate("synthetic-valid-question", ids)
    return validate_scores(ids, scores, score_direction=score_direction)[:5]


def make_trace_template(*, model_id: str, revision: str, query_id: str, candidate_ids: list[str], actual_input: Any, score_contract: str) -> dict[str, Any]:
    validate_slate("synthetic-valid-question", candidate_ids)
    return {
        "contract_version": CONTRACT_VERSION,
        "model_id": model_id,
        "revision": revision,
        "query_id": query_id,
        "candidate_ids": candidate_ids,
        "score_contract": score_contract,
        "actual_model_input_sha256": canonical_hash(actual_input),
        "raw_response_sha256": None,
        "physical_forward_calls": None,
        "cache_reuse": False,
        "cache_provenance": None,
        "gold_in_inference_input": False,
        "status": "NOT_EXECUTED",
    }
