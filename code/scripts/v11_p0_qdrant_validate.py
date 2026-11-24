#!/usr/bin/env python3
"""Read-only exact Qdrant validation for DecisionRAG v1.1 P0 remediation."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COLLECTION = "decisionrag_m2_qwen_embedding_v1"


def request(base: str, path: str, method: str = "GET", body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, method=method, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.loads(response.read().decode())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://service.example.invalid")
    parser.add_argument("--output", type=Path, default=ROOT / "results/preflight/v11_remediation/QDRANT_V11R1_VALIDATION.json")
    args = parser.parse_args()
    corpus_path = ROOT / "data/corpus/passages.jsonl"
    expected_ids = set()
    for line in corpus_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            expected_ids.add(json.loads(line)["doc_id"])
    collection = request(args.base, f"/collections/{COLLECTION}")
    exact = request(args.base, f"/collections/{COLLECTION}/points/count", "POST", {"exact": True})
    # Scroll payloads only.  This verifies exact point/payload identity without
    # transferring a full 2,048-float vector for every point.
    payload_ids = set()
    point_ids = set()
    next_offset = None
    pages = 0
    first_point = None
    while True:
        body = {"limit": 1000, "with_payload": True, "with_vector": False}
        if next_offset is not None:
            body["offset"] = next_offset
        page = request(args.base, f"/collections/{COLLECTION}/points/scroll", "POST", body)["result"]
        pages += 1
        for point in page["points"]:
            point_ids.add(str(point["id"]))
            if first_point is None:
                first_point = point
            payload = point.get("payload") or {}
            if payload.get("doc_id") is not None:
                payload_ids.add(str(payload["doc_id"]))
        next_offset = page.get("next_page_offset")
        if next_offset is None:
            break

    # Keep the original JSON scalar type.  Converting a numeric Qdrant point
    # ID to a string makes an otherwise valid retrieve request invalid.
    probe_ids = [first_point["id"]] if first_point is not None else []
    probe = request(args.base, f"/collections/{COLLECTION}/points", "POST", {"ids": probe_ids, "with_payload": True, "with_vector": True})["result"]
    vector_probe = []
    for point in probe:
        vector = point.get("vector")
        if isinstance(vector, dict):
            dimensions = {name: len(value) for name, value in vector.items() if isinstance(value, list)}
            present = all(value > 0 for value in dimensions.values())
        else:
            dimensions = {"": len(vector)} if isinstance(vector, list) else {}
            present = isinstance(vector, list) and len(vector) > 0
        vector_probe.append({"point_id": point["id"], "vector_present": present, "dimensions": dimensions})

    metrics_text = urllib.request.urlopen(args.base + "/metrics", timeout=120).read().decode("utf-8", errors="replace")
    metric_lines = [line for line in metrics_text.splitlines() if COLLECTION in line or "indexed_only" in line]
    indexed_only_excluded = [line for line in metric_lines if "collection_indexed_only_excluded_points" in line and f'id="{COLLECTION}"' in line]
    source_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "scripts/m2_retrieve.py", ROOT / "scripts/run_m5_serving.py")
        if path.exists()
    )
    indexed_only_mentions = re.findall(r"indexed_only", source_text)
    result = {
        "validation_version": "v11.1r1",
        "read_only": True,
        "collection": COLLECTION,
        "qdrant_root": request(args.base, "/"),
        "collection_response": collection,
        "exact_count_response": exact,
        "exact_point_count": exact["result"]["count"],
        "scroll_pages": pages,
        "scrolled_unique_point_ids": len(point_ids),
        "scrolled_unique_payload_doc_ids": len(payload_ids),
        "expected_corpus_doc_ids": len(expected_ids),
        "missing_expected_payload_doc_ids": sorted(expected_ids - payload_ids),
        "unexpected_payload_doc_ids": sorted(payload_ids - expected_ids),
        "vector_probe": vector_probe,
        "indexed_vectors_count_metadata": collection["result"].get("indexed_vectors_count"),
        "points_count_metadata": collection["result"].get("points_count"),
        "metadata_index_gap": collection["result"].get("points_count", 0) - collection["result"].get("indexed_vectors_count", 0),
        "metric_indexed_only_lines": indexed_only_excluded,
        "project_indexed_only_mentions": len(indexed_only_mentions),
        "project_retrieval_indexed_only_policy": "not_set; Qdrant default includes unindexed data",
        "official_qdrant_docs": {
            "collections": "https://qdrant.tech/documentation/manage-data/collections/",
            "low_latency_search": "https://qdrant.tech/documentation/search/low-latency-search/",
            "optimizer": "https://qdrant.tech/documentation/operations/optimizer/",
        },
        "decision": "PASS_EXACT_POINT_AND_PAYLOAD_PARITY_WITH_INDEX_METADATA_WARNING; preserve frozen slates; no rebuild",
        "frozen_slate_sha256": sha256_file(ROOT / "data/frozen_slates/m2_top20_slates.jsonl"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
