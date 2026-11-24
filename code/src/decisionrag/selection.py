"""Frozen M3 input views, deterministic ranking and gold-blind rescue."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable


POINTWISE_INSTRUCTION = (
    "Does this passage contain factual evidence useful for answering the question, "
    "including an intermediate entity link needed for a multi-hop answer? Answer true "
    "only for actual evidence, not for general topic similarity."
)
GROUPWISE_INSTRUCTION = (
    "Choose the ONE passage in A, B, C, or D that contributes the most useful direct or "
    "intermediate factual evidence for answering the question. A bridge passage can be "
    "valuable even without repeating the question terms."
)
BOOL_CRITERIA = {
    "true": "The passage contains factual evidence useful for answering the question.",
    "false": "The passage does not contain factual evidence useful for answering the question.",
}
GENERIC_TITLES = {"title", "page", "home", "index", "introduction", "overview", "unknown"}


@dataclass(frozen=True)
class CandidateView:
    doc_id: str
    title: str
    full_text: str
    question: str
    question_view: str
    passage_view: str
    selection_text_sha256: str
    original_rank: int
    dataset: str


def sha256_json(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def cap_tokens(tokenizer, text: str, maximum: int) -> tuple[str, int, bool]:
    ids = tokenizer.encode(str(text), add_special_tokens=False)
    clipped = len(ids) > maximum
    return tokenizer.decode(ids[:maximum], skip_special_tokens=True).strip(), len(ids[:maximum]), clipped


def make_views(question: str, candidate: dict, tokenizer, question_limit: int = 64, passage_limit: int = 128) -> CandidateView:
    q_view, _, _ = cap_tokens(tokenizer, question, question_limit)
    passage_source = f"{candidate['title']}\n{candidate['full_text']}"
    p_view, _, _ = cap_tokens(tokenizer, passage_source, passage_limit + len(tokenizer.encode(str(candidate['title']), add_special_tokens=False)))
    # The title is explicitly retained even when its tokenization consumes part of
    # the paragraph budget; all models receive this exact decoded view.
    payload = {"question": q_view, "passage_id": candidate["doc_id"], "title": candidate["title"], "passage": p_view}
    return CandidateView(
        doc_id=candidate["doc_id"], title=candidate["title"], full_text=candidate["full_text"], question=question,
        question_view=q_view, passage_view=p_view, selection_text_sha256=sha256_json(payload),
        original_rank=int(candidate["original_rank"]), dataset=candidate["dataset"],
    )


def pointwise_state(view: CandidateView) -> dict:
    return {"question": view.question_view, "passage_id": view.doc_id, "title": view.title, "passage": view.passage_view}


def pointwise_pair(view: CandidateView) -> tuple[str, str]:
    return view.question_view, f"{view.title}\n{view.passage_view}"


def pointwise_question() -> dict:
    return {"relevance": {"type": "noul", "instructions": POINTWISE_INSTRUCTION, "criteria": BOOL_CRITERIA}}


def groupwise_states(question: str, views: list[CandidateView]) -> tuple[list[dict], dict]:
    if len(views) != 4:
        raise ValueError(f"groupwise requires four views, got {len(views)}")
    choices = {}
    for label, view in zip("ABCD", views):
        choices[label] = {"passage_id": view.doc_id, "title": view.title, "passage": view.passage_view}
    # Keep the quartet text in the typed option criteria only.  Repeating it in
    # state and criteria would consume the base-English Laya 512-token window;
    # the criteria are the official option-bearing representation.
    state = {"question": views[0].question_view}
    question_schema = {"choice": {"type": "choice", "instructions": GROUPWISE_INSTRUCTION, "criteria": {label: f"Passage {label}: {choices[label]['title']} — {choices[label]['passage']}" for label in "ABCD"}}}
    return [state], question_schema


def canonical_title(title: str) -> str:
    value = unicodedata.normalize("NFKC", str(title)).casefold()
    value = re.sub(r"\([^)]*\)", " ", value)
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def searchable_title(title: str) -> str | None:
    value = canonical_title(title)
    if len(value) < 5 or value in GENERIC_TITLES:
        return None
    return value


def normalized_body(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text)).casefold()
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def title_occurs_in_body(title: str, body: str) -> bool:
    title_norm = searchable_title(title)
    if not title_norm:
        return False
    body_norm = normalized_body(body)
    return bool(re.search(r"(?<!\w)" + re.escape(title_norm) + r"(?!\w)", body_norm))


def linked(a: dict, b: dict) -> bool:
    return title_occurs_in_body(a["title"], b["full_text"]) or title_occurs_in_body(b["title"], a["full_text"])


def rank_indices(candidates: list[dict], scores: dict[str, float]) -> list[dict]:
    return sorted(candidates, key=lambda c: (-float(scores[c["doc_id"]]), int(c["original_rank"]), c["doc_id"]))


def rescue_top5(candidates: list[dict], scores: dict[str, float], top5_ids: Iterable[str]) -> dict:
    """Apply exactly the frozen one-swap title-link bridge rescue.

    This function accepts only public candidate bodies and model scores.  It has
    no access to gold labels, answers, hop labels or question records.
    """
    by_id = {c["doc_id"]: c for c in candidates}
    selected = [by_id[x] for x in top5_ids]
    ranked = rank_indices(candidates, scores)
    selected_ids = {c["doc_id"] for c in selected}
    eligible = []
    edges = []
    for candidate in ranked:
        if candidate["doc_id"] in selected_ids or candidate["original_rank"] <= 5:
            continue
        neighbors = [anchor for anchor in selected if linked(candidate, anchor)]
        if neighbors:
            eligible.append((candidate, neighbors))
            for anchor in neighbors:
                edges.append({"candidate_id": candidate["doc_id"], "anchor_id": anchor["doc_id"]})
    if not eligible:
        return {"selected_ids": [c["doc_id"] for c in selected], "swapped": False, "eligible_edges": edges, "reason": "no_connected_outsider", "removed_id": None, "inserted_id": None, "anchor_id": None}
    candidate, neighbors = sorted(eligible, key=lambda item: (-len(item[1]), -float(scores[item[0]["doc_id"]]), item[0]["doc_id"]))[0]
    anchor = sorted(neighbors, key=lambda c: (-float(scores[c["doc_id"]]), int(c["original_rank"]), c["doc_id"]))[0]
    removable = [c for c in selected if c["doc_id"] != anchor["doc_id"]]
    removed = sorted(removable, key=lambda c: (float(scores[c["doc_id"]]), c["doc_id"]))[0]
    output = [c["doc_id"] for c in selected if c["doc_id"] != removed["doc_id"]] + [candidate["doc_id"]]
    return {"selected_ids": output, "swapped": True, "eligible_edges": edges, "reason": "connected_outsider", "removed_id": removed["doc_id"], "inserted_id": candidate["doc_id"], "anchor_id": anchor["doc_id"]}


def deterministic_quartets(candidates: list[dict], query_id: str, seed: int) -> list[list[dict]]:
    import random
    values = list(candidates)
    digest = hashlib.sha256(f"{query_id}:{seed}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    rng.shuffle(values)
    return [values[start : start + 4] for start in range(0, 20, 4)]
