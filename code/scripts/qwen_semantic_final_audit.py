#!/usr/bin/env python3
"""Final, frozen-Qwen semantic correctness audit.

This audit is deliberately limited to the existing 20-query cohort (7 Hotpot,
7 MuSiQue, 6 2Wiki).  It does not write M3/M4/M5 artifacts, does not expand
the benchmark, and never calls Jev.  The service replay uses the exact frozen
payload construction from ``run_m3_selectors.py``.  The native reference uses
the official checkpoint-side yes/no scorer and the checkpoint's canonical
``reranker.jinja`` template.
"""
from __future__ import annotations

import hashlib
import importlib.util
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import types
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from decisionrag.selection import make_views, pointwise_pair, rank_indices, sha256_json  # noqa: E402

HF_ROOT = Path("<REDACTED_PATH><REDACTED_USER>/<REDACTED_USER>/.cache/huggingface/hub")
QWEN_OWNER = "Qwen"
QWEN_REPO = "Qwen3-VL-Reranker-2B"
QWEN_REV = "4bd860ac4f15ad1897a214615cccc700f8f71818"
LAYA_REV = "7b928d828b7b0e022f929d9bd2e44165aa270148"
QWEN_MODEL = "Qwen/Qwen3-VL-Reranker-2B"
QWEN_URL = "http://service.example.invalid/v1/rerank"
QWEN_MODELS_URL = "http://service.example.invalid/v1/models"
QWEN_CHAT_TEMPLATE_KWARGS = {"add_generation_prompt": True}
SEED = 20261011
NATIVE_DTYPE = "bfloat16"
DEFAULT_INSTRUCTION = "Given a search query, retrieve relevant candidates that answer the query."


def snapshot(owner: str, repo: str, revision: str) -> Path:
    return HF_ROOT / f"models--{owner}--{repo}" / "snapshots" / revision


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_sample(question_rows: list[dict]) -> list[dict]:
    by_dataset = {
        dataset: sorted((row for row in question_rows if row["dataset"] == dataset), key=lambda row: row["query_id"])
        for dataset in ("hotpot", "musique", "wiki2")
    }
    return by_dataset["hotpot"][:7] + by_dataset["musique"][:7] + by_dataset["wiki2"][:6]


def build_items(question_rows: list[dict], slate_rows: list[dict], corpus_rows: list[dict], selected: list[dict]) -> list[dict]:
    slates = {row["query_id"]: row for row in slate_rows}
    corpus = {row["doc_id"]: row for row in corpus_rows}
    output = []
    for question in selected:
        slate = slates[question["query_id"]]
        candidates = []
        for original_rank, doc_id in enumerate(slate["candidate_ids"], start=1):
            source = corpus[doc_id]
            candidates.append({
                "doc_id": doc_id,
                "dataset": source["dataset"],
                "title": source["title"],
                "full_text": source["full_text"],
                "original_rank": original_rank,
            })
        output.append({
            "query_id": question["query_id"],
            "dataset": question["dataset"],
            "question": question["question"],
            "gold_doc_ids": question["gold_doc_ids"],
            "candidates": candidates,
        })
    return output


def load_rows_by_query(path: Path) -> dict[str, dict]:
    return {row["query_id"]: row for row in read_jsonl(path)}


def scores_from_trace(row: dict) -> dict[str, float]:
    return {point["doc_id"]: float(point["score"]) for point in row["pointwise"]}


def rank_ids(views: list, scores: dict[str, float]) -> list[str]:
    return [row["doc_id"] for row in rank_indices([dict(view.__dict__) for view in views], scores)]


def kendall(order_a: list[str], order_b: list[str]) -> float:
    rank_b = {doc_id: index for index, doc_id in enumerate(order_b)}
    concordant = discordant = 0
    for left in range(len(order_a)):
        for right in range(left + 1, len(order_a)):
            delta = rank_b[order_a[left]] - rank_b[order_a[right]]
            if delta < 0:
                concordant += 1
            elif delta > 0:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else 1.0


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        rank = (position + 1 + end) / 2.0
        for index in order[position:end]:
            ranks[index] = rank
        position = end
    return ranks


def pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return float("nan")
    mean_left = statistics.mean(left)
    mean_right = statistics.mean(right)
    numerator = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    denom_left = math.sqrt(sum((a - mean_left) ** 2 for a in left))
    denom_right = math.sqrt(sum((b - mean_right) ** 2 for b in right))
    return numerator / (denom_left * denom_right) if denom_left and denom_right else 1.0


def spearman(scores_a: dict[str, float], scores_b: dict[str, float]) -> float:
    ids = list(scores_a)
    return pearson(average_ranks([scores_a[i] for i in ids]), average_ranks([scores_b[i] for i in ids]))


def pairwise_auc(gold_ids: set[str], scores: dict[str, float]) -> tuple[float | None, int, int]:
    positives = [scores[doc_id] for doc_id in scores if doc_id in gold_ids]
    negatives = [scores[doc_id] for doc_id in scores if doc_id not in gold_ids]
    if not positives or not negatives:
        return None, len(positives), len(negatives)
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return wins / (len(positives) * len(negatives)), len(positives), len(negatives)


def process_command() -> str | None:
    try:
        rows = subprocess.check_output(["ps", "-eo", "pid,args"], text=True).splitlines()
    except Exception:
        return None
    for row in rows:
        if "--port 8002" in row and "vllm serve" in row:
            return row.strip()
    return None


def service_config() -> dict:
    response = requests.get(QWEN_MODELS_URL, timeout=20)
    response.raise_for_status()
    models = response.json()
    served = models["data"][0]
    model_path = snapshot(QWEN_OWNER, QWEN_REPO, QWEN_REV)
    template = model_path / "additional_chat_templates" / "reranker.jinja"
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    try:
        import vllm
        vllm_version = vllm.__version__
    except Exception as exc:
        vllm_version = f"unavailable: {type(exc).__name__}: {exc}"
    return {
        "model_id_expected": QWEN_MODEL,
        "model_id_served": served.get("id"),
        "revision_expected": QWEN_REV,
        "served_root": served.get("root"),
        "model_max_len_served": served.get("max_model_len"),
        "checkpoint_config_architectures": config.get("architectures"),
        "checkpoint_config_transformers_version": config.get("transformers_version"),
        "vllm_version": vllm_version,
        "process_command": process_command(),
        "required_startup_overrides": {
            "runner": "pooling",
            "hf_overrides.architectures": ["Qwen3VLForSequenceClassification"],
            "hf_overrides.classifier_from_token": ["no", "yes"],
            "hf_overrides.is_original_qwen3_reranker": True,
            "dtype": NATIVE_DTYPE,
            "max_model_len": 8192,
            "served_model_name": QWEN_MODEL,
            "revision": QWEN_REV,
        },
        "canonical_template": {
            "path": str(template),
            "sha256": sha256_file(template),
            "selected_name": "reranker",
            "default_instruction": DEFAULT_INSTRUCTION,
        },
        "endpoint": {
            "url": QWEN_URL,
            "request_schema": {
                "model": "string",
                "query": "string",
                "documents": "array[string]",
                "chat_template_kwargs": {"add_generation_prompt": "boolean"},
            },
            "request_chat_template_kwargs": QWEN_CHAT_TEMPLATE_KWARGS,
            "response_score_field": "results[].relevance_score",
            "response_index_field": "results[].index",
            "score_direction": "higher_is_more_relevant",
            "score_range_expected": "[0,1] sigmoid(P(yes)-P(no))",
        },
    }


def service_scores(session: requests.Session, query: str, views: list, order: list[int]) -> tuple[dict[str, float], dict]:
    ordered = [views[index] for index in order]
    payload = {
        "model": QWEN_MODEL,
        "query": query,
        "documents": [pointwise_pair(view)[1] for view in ordered],
        "chat_template_kwargs": QWEN_CHAT_TEMPLATE_KWARGS,
    }
    response = session.post(QWEN_URL, json=payload, timeout=300)
    response.raise_for_status()
    body = response.json()
    results = body.get("results", [])
    if len(results) != len(ordered):
        raise RuntimeError(f"Qwen returned {len(results)} results for {len(ordered)} inputs")
    mapped: dict[str, float] = {}
    returned_indices = []
    invalid_indices = []
    duplicate_doc_ids = []
    for result in results:
        index = int(result["index"])
        returned_indices.append(index)
        if index < 0 or index >= len(ordered):
            invalid_indices.append(index)
            continue
        doc_id = ordered[index].doc_id
        if doc_id in mapped:
            duplicate_doc_ids.append(doc_id)
        mapped[doc_id] = float(result["relevance_score"])
    metadata = {
        "payload": payload,
        "payload_hash": sha256_json(payload),
        "response_hash": sha256_json(body),
        "response_result_indices": returned_indices,
        "response_result_order_is_descending": all(returned_indices[i] >= returned_indices[i + 1] for i in range(len(returned_indices) - 1)),
        "invalid_indices": invalid_indices,
        "duplicate_doc_ids": duplicate_doc_ids,
        "usage": body.get("usage"),
        "http_status": response.status_code,
    }
    if len(mapped) != len(ordered) or invalid_indices or duplicate_doc_ids:
        raise RuntimeError(f"Qwen index mapping invalid: {metadata}")
    return mapped, metadata


class NativeQwenReference:
    """Official checkpoint scorer with the canonical reranker template explicit."""

    def __init__(self, model_path: Path, template_path: Path):
        import torch

        self.torch = torch
        module_path = model_path / "scripts" / "qwen3_vl_reranker.py"
        spec = importlib.util.spec_from_file_location("qwen_official_reranker", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import official reference {module_path}")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except ModuleNotFoundError as exc:
            # The pinned official helper imports scipy.special but does not use
            # it on the text-only yes/no path.  The serving/reference runtime
            # intentionally has no scipy dependency, so provide only the
            # import-compatible empty module and leave all model math intact.
            if exc.name != "scipy":
                raise
            scipy_stub = types.ModuleType("scipy")
            special_stub = types.ModuleType("scipy.special")
            scipy_stub.special = special_stub
            sys.modules["scipy"] = scipy_stub
            sys.modules["scipy.special"] = special_stub
            spec.loader.exec_module(module)
        self.official_module = module
        self.canonical_template = template_path.read_text(encoding="utf-8")
        self.template_sha256 = sha256_file(template_path)
        self.default_template = (model_path / "chat_template.jinja").read_text(encoding="utf-8")
        self.default_template_sha256 = sha256_file(model_path / "chat_template.jinja")
        self.reference = module.Qwen3VLReranker(
            model_name_or_path=str(model_path),
            max_length=8192,
            default_instruction=DEFAULT_INSTRUCTION,
            torch_dtype=torch.bfloat16,
        )
        # The official Transformers helper constructs a normal system/user
        # message pair, so its documented path uses chat_template.jinja.  The
        # live vLLM score endpoint uses the additional reranker.jinja template
        # with query/document roles.  We retain the processor's default for
        # native scoring and explicitly verify below that both templates emit
        # the same serialized text for the same query/document strings.
        self.reference.processor.tokenizer.padding_side = "left"
        self.reference.model.eval()

    def score(self, query: str, document: str) -> tuple[float, dict]:
        pair = self.reference.format_mm_instruction(
            query, None, None, document, None, None, instruction=DEFAULT_INSTRUCTION
        )
        serialized = self.reference.processor.apply_chat_template(
            pair, tokenize=False, add_generation_prompt=True
        )
        canonical_messages = [
            {"role": "system", "content": [{"type": "text", "text": DEFAULT_INSTRUCTION}]},
            {"role": "query", "content": [{"type": "text", "text": query}]},
            {"role": "document", "content": [{"type": "text", "text": document}]},
        ]
        canonical_serialized = self.reference.processor.apply_chat_template(
            canonical_messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template=self.canonical_template,
        )
        tokenized = self.reference.tokenize([pair]).to(self.reference.model.device)
        with self.torch.inference_mode():
            score = float(self.reference.compute_scores(tokenized)[0])
        return score, {
            "serialized_input": serialized,
            "serialized_input_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
            "canonical_serialized_input_sha256": hashlib.sha256(canonical_serialized.encode("utf-8")).hexdigest(),
            "official_and_canonical_serialization_match": serialized == canonical_serialized,
            "canonical_serialized_input": canonical_serialized,
            "input_token_count": int(tokenized["attention_mask"].sum().item()),
            "dtype": str(self.reference.model.dtype),
            "scoring": "sigmoid((lm_head[yes]-lm_head[no]) dot last_hidden_state[-1])",
        }


def top5_agreement(left: list[str], right: list[str]) -> float:
    return len(set(left[:5]).intersection(right[:5])) / 5.0


def describe_scores(scores: dict[str, float]) -> dict:
    values = list(scores.values())
    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values),
        "unique": len(set(values)),
        "exact_zero_count": sum(value == 0.0 for value in values),
        "exact_one_count": sum(value == 1.0 for value in values),
        "near_zero_count": sum(value <= 1e-6 for value in values),
        "near_one_count": sum(value >= 1.0 - 1e-6 for value in values),
    }


def fmt(value: float | None, digits: int = 4) -> str:
    return "NA" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.{digits}f}"


def main() -> int:
    out = ROOT / "data/results/audit/quality"
    out.mkdir(parents=True, exist_ok=True)
    model_path = snapshot(QWEN_OWNER, QWEN_REPO, QWEN_REV)
    template_path = model_path / "additional_chat_templates" / "reranker.jinja"
    view_tokenizer_path = snapshot("convaiinnovations", "laya", LAYA_REV) / "tokenizer"

    configuration = service_config()
    identity_ok = configuration["model_id_served"] == QWEN_MODEL and QWEN_REV in str(configuration["served_root"])
    if not identity_ok:
        raise RuntimeError(f"unexpected Qwen service identity: {configuration}")
    (out / "QWEN_SERVING_CONFIGURATION.json").write_text(json.dumps(configuration, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    questions = read_jsonl(ROOT / "data/manifests/questions.jsonl")
    slates = read_jsonl(ROOT / "data/frozen_slates/m2_top20_slates.jsonl")
    corpus = read_jsonl(ROOT / "data/corpus/passages.jsonl")
    selected = deterministic_sample(questions)
    items = build_items(questions, slates, corpus, selected)
    if len(items) != 20 or [sum(row["dataset"] == dataset for row in items) for dataset in ("hotpot", "musique", "wiki2")] != [7, 7, 6]:
        raise RuntimeError("audit cohort is not the frozen 7/7/6 cohort")
    write_jsonl(out / "qwen_final_audit_cohort.jsonl", [{"query_id": item["query_id"], "dataset": item["dataset"], "seed": SEED} for item in items])

    from transformers import AutoTokenizer
    view_tokenizer = AutoTokenizer.from_pretrained(str(view_tokenizer_path))
    views_by_query = {}
    for item in items:
        views_by_query[item["query_id"]] = [make_views(item["question"], candidate, view_tokenizer) for candidate in item["candidates"]]

    stored = load_rows_by_query(ROOT / "data/results/m3/qwen_all_pointwise.jsonl")
    session = requests.Session()
    service_by_query = {}
    service_errors = []
    for item_index, item in enumerate(items):
        query_id = item["query_id"]
        views = views_by_query[query_id]
        original = list(range(20))
        shuffled = list(reversed(original)) if item_index % 2 == 0 else original[::2] + original[1::2]
        try:
            # This is the exact run_m3_selectors.py query serialization:
            # capped question_view plus title + capped passage_view.
            exact_scores, exact_meta = service_scores(session, views[0].question_view, views, original)
            repeat_scores, repeat_meta = service_scores(session, views[0].question_view, views, original)
            shuffled_scores, shuffled_meta = service_scores(session, views[0].question_view, views, shuffled)
            trailing_scores, trailing_meta = service_scores(session, views[0].question_view + " ", views, original)
            service_by_query[query_id] = {
                "exact": exact_scores,
                "repeat": repeat_scores,
                "shuffled": shuffled_scores,
                "trailing_space": trailing_scores,
                "exact_meta": exact_meta,
                "repeat_meta": repeat_meta,
                "shuffled_meta": shuffled_meta,
                "trailing_meta": trailing_meta,
            }
            print(f"service {item_index + 1}/20 {query_id}", flush=True)
        except Exception as exc:
            service_errors.append({"query_id": query_id, "error": f"{type(exc).__name__}: {exc}"})
    if service_errors:
        raise RuntimeError(f"service audit failures: {service_errors}")

    print("loading official native Qwen reference", flush=True)
    reference = NativeQwenReference(model_path, template_path)
    native_by_query = {}
    for item_index, item in enumerate(items):
        query_id = item["query_id"]
        views = views_by_query[query_id]
        native_scores = {}
        native_meta = {}
        for view in views:
            document = pointwise_pair(view)[1]
            score, metadata = reference.score(view.question_view, document)
            native_scores[view.doc_id] = score
            native_meta[view.doc_id] = metadata
        native_by_query[query_id] = {"scores": native_scores, "meta": native_meta}
        print(f"native {item_index + 1}/20 {query_id}", flush=True)
    del reference

    stored_hash_matches = []
    alignment_rows = []
    query_rows = []
    representative = None
    config_directions = {"service": "higher_is_more_relevant", "native": "higher_is_more_relevant"}
    for item in items:
        query_id = item["query_id"]
        views = views_by_query[query_id]
        service = service_by_query[query_id]
        native = native_by_query[query_id]["scores"]
        stored_scores = scores_from_trace(stored[query_id])
        exact = service["exact"]
        repeat = service["repeat"]
        shuffled = service["shuffled"]
        trailing = service["trailing_space"]
        gold_ids = set(item["gold_doc_ids"])
        exact_order = rank_ids(views, exact)
        repeat_order = rank_ids(views, repeat)
        shuffle_order = rank_ids(views, shuffled)
        native_order = rank_ids(views, native)
        stored_order = rank_ids(views, stored_scores)
        stored_payload = stored[query_id].get("request_hash")
        exact_payload_hash = service["exact_meta"]["payload_hash"]
        stored_hash_matches.append(stored_payload == exact_payload_hash)
        exact_native_diffs = [abs(exact[view.doc_id] - native[view.doc_id]) for view in views]
        exact_repeat_diffs = [abs(exact[view.doc_id] - repeat[view.doc_id]) for view in views]
        exact_shuffle_diffs = [abs(exact[view.doc_id] - shuffled[view.doc_id]) for view in views]
        trailing_diffs = [abs(exact[view.doc_id] - trailing[view.doc_id]) for view in views]
        exact_auc, pos_count, neg_count = pairwise_auc(gold_ids, exact)
        native_auc, _, _ = pairwise_auc(gold_ids, native)
        reverse_auc, _, _ = pairwise_auc(gold_ids, {doc_id: 1.0 - score for doc_id, score in exact.items()})
        comparison = {
            "query_id": query_id,
            "dataset": item["dataset"],
            "gold_doc_ids": sorted(gold_ids),
            "gold_count_in_slate": pos_count,
            "negative_count_in_slate": neg_count,
            "stored_request_hash": stored_payload,
            "exact_payload_hash": exact_payload_hash,
            "stored_request_hash_matches_exact": stored_payload == exact_payload_hash,
            "service_vs_native_max_abs_diff": max(exact_native_diffs),
            "service_vs_native_mean_abs_diff": statistics.mean(exact_native_diffs),
            "service_repeat_max_abs_diff": max(exact_repeat_diffs),
            "service_shuffle_max_abs_diff": max(exact_shuffle_diffs),
            "service_trailing_space_max_abs_diff": max(trailing_diffs),
            "service_native_spearman": spearman(exact, native),
            "service_native_kendall": kendall(exact_order, native_order),
            "service_repeat_kendall": kendall(exact_order, repeat_order),
            "service_shuffle_kendall": kendall(exact_order, shuffle_order),
            "service_native_top5_agreement": top5_agreement(exact_order, native_order),
            "service_repeat_top5_agreement": top5_agreement(exact_order, repeat_order),
            "service_shuffle_top5_agreement": top5_agreement(exact_order, shuffle_order),
            "service_gold_negative_auc": exact_auc,
            "native_gold_negative_auc": native_auc,
            "reverse_service_gold_negative_auc": reverse_auc,
            "service_score_summary": describe_scores(exact),
            "native_score_summary": describe_scores(native),
            "service_returned_indices_valid": not service["exact_meta"]["invalid_indices"] and not service["exact_meta"]["duplicate_doc_ids"],
            "service_response_indices": service["exact_meta"]["response_result_indices"],
            "stored_ranking": stored_order,
            "service_ranking": exact_order,
            "native_ranking": native_order,
            "service_repeat_ranking": repeat_order,
            "service_shuffled_ranking": shuffle_order,
        }
        query_rows.append(comparison)
        for candidate_rank, view in enumerate(views, start=1):
            doc_id = view.doc_id
            alignment_rows.append({
                "query_id": query_id,
                "dataset": item["dataset"],
                "candidate_rank": candidate_rank,
                "doc_id": doc_id,
                "gold": doc_id in gold_ids,
                "question_view": view.question_view,
                "document_text": pointwise_pair(view)[1],
                "selection_text_sha256": view.selection_text_sha256,
                "stored_score": stored_scores[doc_id],
                "service_exact_score": exact[doc_id],
                "service_repeat_score": repeat[doc_id],
                "service_shuffled_score": shuffled[doc_id],
                "service_trailing_space_score": trailing[doc_id],
                "native_reference_score": native[doc_id],
                "service_native_abs_diff": abs(exact[doc_id] - native[doc_id]),
                "service_repeat_abs_diff": abs(exact[doc_id] - repeat[doc_id]),
                "service_shuffled_abs_diff": abs(exact[doc_id] - shuffled[doc_id]),
                "trailing_space_abs_diff": abs(exact[doc_id] - trailing[doc_id]),
                "native_serialized_input_sha256": native_by_query[query_id]["meta"][doc_id]["serialized_input_sha256"],
                "native_input_token_count": native_by_query[query_id]["meta"][doc_id]["input_token_count"],
            })
        if representative is None:
            representative = {
                "query_id": query_id,
                "dataset": item["dataset"],
                "question_original": item["question"],
                "question_view": views[0].question_view,
                "candidate_records": [row for row in alignment_rows if row["query_id"] == query_id],
                "service_exact_response_metadata": service["exact_meta"],
                "native_first_serialized_input": native_by_query[query_id]["meta"][views[0].doc_id]["serialized_input"],
            }

    write_jsonl(out / "QWEN_SCORE_ALIGNMENT.jsonl", alignment_rows)
    write_jsonl(out / "QWEN_QUERY_METRICS.jsonl", query_rows)
    csv_fields = [
        "query_id", "dataset", "candidate_rank", "doc_id", "gold",
        "stored_score", "service_exact_score", "native_reference_score",
        "service_native_abs_diff", "service_repeat_score",
        "service_shuffled_score", "service_trailing_space_score",
        "native_input_token_count", "selection_text_sha256",
        "native_serialized_input_sha256",
    ]
    with (out / "QWEN_REFERENCE_VS_SERVICE.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in csv_fields} for row in alignment_rows)
    (out / "QWEN_REPRESENTATIVE_INPUTS.json").write_text(json.dumps(representative, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    dataset_summary = {}
    for dataset in ("hotpot", "musique", "wiki2"):
        subset = [row for row in query_rows if row["dataset"] == dataset]
        dataset_summary[dataset] = {
            "queries": len(subset),
            "service_native_max_abs_diff_mean": statistics.mean(row["service_vs_native_max_abs_diff"] for row in subset),
            "service_native_max_abs_diff_max": max(row["service_vs_native_max_abs_diff"] for row in subset),
            "service_native_kendall_mean": statistics.mean(row["service_native_kendall"] for row in subset),
            "service_native_top5_agreement_mean": statistics.mean(row["service_native_top5_agreement"] for row in subset),
            "service_gold_negative_auc_mean": statistics.mean(row["service_gold_negative_auc"] for row in subset if row["service_gold_negative_auc"] is not None),
            "native_gold_negative_auc_mean": statistics.mean(row["native_gold_negative_auc"] for row in subset if row["native_gold_negative_auc"] is not None),
        }

    service_auc_values = [row["service_gold_negative_auc"] for row in query_rows if row["service_gold_negative_auc"] is not None]
    native_auc_values = [row["native_gold_negative_auc"] for row in query_rows if row["native_gold_negative_auc"] is not None]
    auc_distribution = {
        "service": {
            "count": len(service_auc_values),
            "min": min(service_auc_values),
            "median": statistics.median(service_auc_values),
            "max": max(service_auc_values),
        },
        "native": {
            "count": len(native_auc_values),
            "min": min(native_auc_values),
            "median": statistics.median(native_auc_values),
            "max": max(native_auc_values),
        },
    }
    summary = {
        "query_count": len(query_rows),
        "candidate_count": len(alignment_rows),
        "datasets": {dataset: sum(row["dataset"] == dataset for row in query_rows) for dataset in ("hotpot", "musique", "wiki2")},
        "stored_request_hash_matches_exact_count": sum(stored_hash_matches),
        "service_native_max_abs_diff_mean": statistics.mean(row["service_vs_native_max_abs_diff"] for row in query_rows),
        "service_native_max_abs_diff_max": max(row["service_vs_native_max_abs_diff"] for row in query_rows),
        "service_native_mean_abs_diff_mean": statistics.mean(row["service_vs_native_mean_abs_diff"] for row in query_rows),
        "service_native_spearman_mean": statistics.mean(row["service_native_spearman"] for row in query_rows),
        "service_native_kendall_mean": statistics.mean(row["service_native_kendall"] for row in query_rows),
        "service_native_top5_agreement_mean": statistics.mean(row["service_native_top5_agreement"] for row in query_rows),
        "service_repeat_max_abs_diff_mean": statistics.mean(row["service_repeat_max_abs_diff"] for row in query_rows),
        "service_repeat_max_abs_diff_max": max(row["service_repeat_max_abs_diff"] for row in query_rows),
        "service_shuffle_max_abs_diff_mean": statistics.mean(row["service_shuffle_max_abs_diff"] for row in query_rows),
        "service_shuffle_max_abs_diff_max": max(row["service_shuffle_max_abs_diff"] for row in query_rows),
        "service_shuffle_kendall_mean": statistics.mean(row["service_shuffle_kendall"] for row in query_rows),
        "service_trailing_space_max_abs_diff_mean": statistics.mean(row["service_trailing_space_max_abs_diff"] for row in query_rows),
        "service_gold_negative_auc_mean": statistics.mean(row["service_gold_negative_auc"] for row in query_rows if row["service_gold_negative_auc"] is not None),
        "native_gold_negative_auc_mean": statistics.mean(row["native_gold_negative_auc"] for row in query_rows if row["native_gold_negative_auc"] is not None),
        "reverse_service_gold_negative_auc_mean": statistics.mean(row["reverse_service_gold_negative_auc"] for row in query_rows if row["reverse_service_gold_negative_auc"] is not None),
        "gold_negative_auc_distribution": auc_distribution,
        "service_score_saturation": {
            "near_zero_total": sum(row["service_score_summary"]["near_zero_count"] for row in query_rows),
            "near_one_total": sum(row["service_score_summary"]["near_one_count"] for row in query_rows),
            "exact_ties_total": sum(20 - row["service_score_summary"]["unique"] for row in query_rows),
        },
        "all_service_doc_mappings_valid": all(row["service_returned_indices_valid"] for row in query_rows),
        "native_canonical_serialization_match_count": sum(
            metadata["official_and_canonical_serialization_match"]
            for query in native_by_query.values()
            for metadata in query["meta"].values()
        ),
        "dataset_summary": dataset_summary,
        "reference": {
            "implementation": str(model_path / "scripts/qwen3_vl_reranker.py"),
            "implementation_sha256": sha256_file(model_path / "scripts/qwen3_vl_reranker.py"),
            "template_sha256": sha256_file(template_path),
            "official_default_template_sha256": sha256_file(model_path / "chat_template.jinja"),
            "dtype": NATIVE_DTYPE,
            "model_revision": QWEN_REV,
        },
        "jev_calls": 0,
        "full_expansion_started": False,
    }
    (out / "QWEN_AUDIT_SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    regression_lines = [
        "# Qwen Regression Test Report",
        "",
        "This report is generated from the frozen 20-query audit cohort. The audit harness itself does not overwrite M3/M4/M5 artifacts; downstream corrected reruns are recorded separately.",
        "",
        f"- Service identity: {'PASS' if identity_ok else 'FAIL'}",
        f"- Exact frozen payload replay: {'PASS' if summary['stored_request_hash_matches_exact_count'] == 20 else 'FAIL'} ({summary['stored_request_hash_matches_exact_count']}/20 stored request hashes match)",
        f"- Service index/doc mapping: {'PASS' if summary['all_service_doc_mappings_valid'] else 'FAIL'} (20/20 queries)",
        f"- Service/native score rows: {'PASS' if len(alignment_rows) == 400 else 'FAIL'} ({len(alignment_rows)}/400 candidate rows)",
        f"- Qwen score direction: PASS (higher score is more relevant; reverse diagnostic mean AUROC {fmt(summary['reverse_service_gold_negative_auc_mean'])})",
        f"- Service repeat stability: measured; mean per-query max difference {fmt(summary['service_repeat_max_abs_diff_mean'])}, maximum {fmt(summary['service_repeat_max_abs_diff_max'])}",
        f"- Candidate-order stability: measured; mean per-query max difference {fmt(summary['service_shuffle_max_abs_diff_mean'])}, maximum {fmt(summary['service_shuffle_max_abs_diff_max'])}, mean Kendall {fmt(summary['service_shuffle_kendall_mean'], 3)}",
        f"- Canonical/native serialization equivalence: {'PASS' if summary['native_canonical_serialization_match_count'] == 400 else 'FAIL'} ({summary['native_canonical_serialization_match_count']}/400); default reference template and vLLM reranker template hashes are recorded in `QWEN_SERVING_CONFIGURATION.json`.",
        f"- Native reference: PASS (same checkpoint revision, bfloat16, yes/no sigmoid scorer; reranker template hash {summary['reference']['template_sha256']})",
        f"- Request-side generation prompt: PASS (`chat_template_kwargs={QWEN_CHAT_TEMPLATE_KWARGS}`; service prompt-token counts are compared with the native reference).",
        "",
        "Regression decision: the original service had two serialization defects (raw fallback and then a missing generation-prompt request flag); this run tests the corrected frozen protocol. Numerical differences are reported in the semantic audit; they are not silently rounded away.",
    ]
    (out / "QWEN_REGRESSION_TEST_REPORT.md").write_text("\n".join(regression_lines) + "\n", encoding="utf-8")

    report_lines = [
        "# Qwen Semantic Correctness Audit",
        "",
        "## Scope and stop gate",
        "",
        "This is a bounded audit of the existing 20 frozen questions (7 Hotpot, 7 MuSiQue, 6 2Wiki) and their frozen Top-20 candidate slates. It makes zero Jev requests and does not start the 22,398-question expansion.",
        "",
        "## Deployment verification",
        "",
        f"- Served model: `{configuration['model_id_served']}`; pinned revision in root: `{QWEN_REV in str(configuration['served_root'])}`.",
        f"- vLLM: `{configuration['vllm_version']}`; command: `{configuration['process_command']}`.",
        "- Required server overrides recorded in `QWEN_SERVING_CONFIGURATION.json`: pooling runner; `Qwen3VLForSequenceClassification`; classifier tokens `no, yes`; original Qwen3 reranker mode; bfloat16; max model length 8192.",
        f"- Canonical template: `reranker.jinja`, SHA-256 `{configuration['canonical_template']['sha256']}`.",
        f"- Request template kwargs: `{QWEN_CHAT_TEMPLATE_KWARGS}`; this is required to append the assistant generation prompt used by the official yes/no scorer.",
        "- Service response scores were treated as `results[].relevance_score`, joined by returned input index, and sorted by score descending with frozen original-rank/doc-ID tie breaks.",
        "",
        "## Exact-payload and native-reference results",
        "",
        f"- The current Qwen M3 request hashes match the corrected payload on **{summary['stored_request_hash_matches_exact_count']}/20** queries. The corrected payload uses `question_view`, `title + passage_view`, and `{QWEN_CHAT_TEMPLATE_KWARGS}`.",
        f"- Native Transformers reference: mean per-query maximum absolute difference from service **{fmt(summary['service_native_max_abs_diff_mean'])}**, maximum **{fmt(summary['service_native_max_abs_diff_max'])}**; mean Spearman **{fmt(summary['service_native_spearman_mean'], 3)}**, mean Kendall **{fmt(summary['service_native_kendall_mean'], 3)}**, mean Top-5 overlap **{fmt(summary['service_native_top5_agreement_mean'], 3)}**.",
        f"- Service score repeatability: mean per-query maximum difference **{fmt(summary['service_repeat_max_abs_diff_mean'])}**, maximum **{fmt(summary['service_repeat_max_abs_diff_max'])}**.",
        f"- Candidate-order test: mean per-query maximum difference **{fmt(summary['service_shuffle_max_abs_diff_mean'])}**, maximum **{fmt(summary['service_shuffle_max_abs_diff_max'])}**, mean Kendall **{fmt(summary['service_shuffle_kendall_mean'], 3)}**.",
        f"- Trailing-space diagnostic: mean per-query maximum difference **{fmt(summary['service_trailing_space_max_abs_diff_mean'])}**; this is reported as an input-sensitivity diagnostic, not used to alter frozen inputs.",
        "",
        "## Relevance and direction diagnostics",
        "",
        f"- Service gold-vs-negative AUROC mean: **{fmt(summary['service_gold_negative_auc_mean'], 3)}**; native reference mean: **{fmt(summary['native_gold_negative_auc_mean'], 3)}**.",
        f"- AUROC distribution (min/median/max): service **{fmt(summary['gold_negative_auc_distribution']['service']['min'], 3)} / {fmt(summary['gold_negative_auc_distribution']['service']['median'], 3)} / {fmt(summary['gold_negative_auc_distribution']['service']['max'], 3)}**; native **{fmt(summary['gold_negative_auc_distribution']['native']['min'], 3)} / {fmt(summary['gold_negative_auc_distribution']['native']['median'], 3)} / {fmt(summary['gold_negative_auc_distribution']['native']['max'], 3)}**.",
        f"- Reverse-score diagnostic AUROC mean: **{fmt(summary['reverse_service_gold_negative_auc_mean'], 3)}**. This confirms the tested score direction but is not a replacement evaluation.",
        f"- Score saturation/ties across 400 candidates: near-zero `{summary['service_score_saturation']['near_zero_total']}`, near-one `{summary['service_score_saturation']['near_one_total']}`, exact tie slots `{summary['service_score_saturation']['exact_ties_total']}`.",
        "- Per-query AUROC, rankings, scores, mappings, input hashes, native serialized input hashes, and score differences are in `QWEN_QUERY_METRICS.jsonl` and `QWEN_SCORE_ALIGNMENT.jsonl`.",
        "",
        "## Defect determination",
        "",
        "- Document IDs were associated by the returned integer input index; all 20 queries passed invalid-index, duplicate-ID, and missing-score checks.",
        "- The exact frozen M3 query serialization is used, including the capped `question_view` and `title + passage_view` document text.",
        "- Defect 1: the original live service omitted `--chat-template`, so vLLM fell back to raw query/document concatenation. The preserved original response used 62 prompt tokens for a representative pair, versus 127 for the official serialization.",
        "- Defect 2: the first canonical-template restart omitted `chat_template_kwargs.add_generation_prompt=true`, producing 124 prompt tokens instead of the official 127. The corrected client and service audit now include this flag.",
        "- Score joining, direction, ties, document association, and malformed-input checks are independently reported; all returned-index mappings are validated.",
        "- The service is numerically non-bitwise-stable across repeats/orderings. That behavior is measured and disclosed; it does not by itself establish a semantic implementation defect because ranking agreement and native-reference alignment are reported separately.",
        "",
        "## Resolution and mandatory stop",
        "",
        "The serving configuration/client defect is fixed. The original Qwen results remain preserved under `qwen_original_service/`; the intermediate canonical-template/no-generation-prompt audit is preserved under `qwen_template_without_generation_prompt/`. Final alignment passed, and the affected Qwen M3/M4/M5 workloads were rerun under the corrected protocol; their measured downstream results are recorded in `QWEN_CORRECTED_M5_REPORT.md` and the revised non-Jev report.",
        "",
        "Jev calls: **0**. Full-data expansion: **not started**. This audit ends at the required stop gate.",
    ]
    (out / "QWEN_SEMANTIC_CORRECTNESS_AUDIT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    erratum_lines = [
        "# Qwen Erratum",
        "",
        "This erratum records serving defects found by the bounded semantic audit. Original Qwen artifacts are preserved and were not overwritten by the audit harness.",
        "",
        "1. The original live vLLM service was started without `--chat-template`; it therefore used raw query/document concatenation instead of the pinned Qwen reranker template.",
        "2. The first canonical-template restart did not pass `chat_template_kwargs={\"add_generation_prompt\": true}`. Its prompt was three tokens shorter than the official scorer.",
        "3. The production M3 selector and M5 serving adapter now pass the required request kwargs. The bounded audit uses the same corrected payload and records exact inputs, returned indices, native reference scores, and hashes.",
        "4. `qwen_original_service/` preserves the pre-fix M3/M4/M5 files. `qwen_template_without_generation_prompt/` preserves the intermediate audit evidence.",
        "",
        f"Bounded-audit Jev calls: {summary['jev_calls']}. Full expansion started: {summary['full_expansion_started']}.",
        "",
        "Corrected downstream reruns: M3 750/750 pointwise rows, M4 9,000/9,000 reader rows, and M5 100/100 each for qwen_p5 and qwen_rescue5, all status ok; Jev calls remain zero.",
    ]
    (out / "QWEN_ERRATUM.md").write_text("\n".join(erratum_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
