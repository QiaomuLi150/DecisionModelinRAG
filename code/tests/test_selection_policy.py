from decisionrag.selection import (
    canonical_title,
    deterministic_quartets,
    rescue_top5,
    searchable_title,
    title_occurs_in_body,
)


def candidate(doc_id, title, body, rank):
    return {"doc_id": doc_id, "dataset": "x", "title": title, "full_text": body, "original_rank": rank}


def test_title_normalization_and_generic_rejection():
    assert canonical_title("Ada Lovelace (mathematician)") == "ada lovelace"
    assert searchable_title("page") is None
    assert searchable_title("Ada Lovelace (mathematician)") == "ada lovelace"
    assert title_occurs_in_body("Ada Lovelace", "The work of Ada Lovelace was important.")
    assert not title_occurs_in_body("Ada Lovelace", "An Adalovelace token is not a phrase match.")


def test_quartets_are_deterministic_and_cover_twenty():
    items = [candidate(str(i), f"Title {i}", "body", i) for i in range(1, 21)]
    a = deterministic_quartets(items, "q1", 20261011)
    b = deterministic_quartets(items, "q1", 20261011)
    assert [[x["doc_id"] for x in group] for group in a] == [[x["doc_id"] for x in group] for group in b]
    assert len(a) == 5 and sorted((int(x["doc_id"]) for group in a for x in group)) == list(range(1, 21))


def test_rescue_one_swap_is_gold_blind_and_tie_deterministic():
    items = [candidate("a", "Anchor A", "mentions Bridge B", 1), candidate("b", "Bridge B", "bridge body", 6), candidate("c", "Other C", "other", 2), candidate("d", "Other D", "other", 3), candidate("e", "Other E", "other", 4), candidate("f", "Other F", "other", 5)]
    scores = {key: 1.0 for key in "abcdef"}
    result = rescue_top5(items, scores, ["a", "c", "d", "e", "f"])
    assert result["swapped"] is True
    assert result["inserted_id"] == "b"
    assert result["removed_id"] == "c"
    assert len(result["selected_ids"]) == 5
    assert len(set(result["selected_ids"])) == 5


def test_rescue_has_no_connected_outsider():
    items = [candidate(str(i), f"Title {i}", "isolated", i) for i in range(1, 7)]
    result = rescue_top5(items, {str(i): float(i) for i in range(1, 7)}, ["1", "2", "3", "4", "5"])
    assert result["swapped"] is False
    assert result["reason"] == "no_connected_outsider"
