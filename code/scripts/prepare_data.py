#!/usr/bin/env python3
"""M1: normalize the three released dev corpora and freeze CAL/TEST/debug."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    return re.sub(r"\s+", " ", text).strip()


def norm_key(text: str) -> str:
    return canonical_text(text).casefold()


def passage_id(dataset: str, title: str, text: str) -> str:
    digest = sha256_bytes((norm_key(title) + "\n" + canonical_text(text)).encode("utf-8"))[:24]
    return f"{dataset}:{digest}"


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def hotpot_rows(path: Path):
    for raw in pq.read_table(path).to_pylist():
        context = raw["context"]
        titles = context["title"]
        sentences = context["sentences"]
        sf = raw["supporting_facts"]
        supporting = list(zip(sf["title"], sf["sent_id"]))
        passages = [(str(title), " ".join(str(sentence) for sentence in sentence_list), list(sentence_list)) for title, sentence_list in zip(titles, sentences)]
        yield {"id": str(raw["id"]), "question": str(raw["question"]), "answer": str(raw["answer"]), "type": str(raw.get("type") or "unknown"), "hop_count": None, "answer_aliases": [], "passages": passages, "supporting": supporting, "source": "hotpotqa/hotpot_qa:distractor/validation"}


def musique_rows(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        prefix = str(raw["id"]).split("__", 1)[0]
        hop_count = int(prefix.replace("hop", "")) if prefix.endswith("hop") and prefix[:-3].isdigit() else None
        passages = [(str(p["title"]), str(p["paragraph_text"]), [str(p["paragraph_text"])]) for p in raw["paragraphs"]]
        supporting = [(str(p["title"]), None) for p in raw["paragraphs"] if p.get("is_supporting")]
        supporting_indices = [index for index, paragraph in enumerate(raw["paragraphs"]) if paragraph.get("is_supporting")]
        yield {"id": str(raw["id"]), "question": str(raw["question"]), "answer": str(raw["answer"]), "type": f"{hop_count}hop" if hop_count else "unknown", "hop_count": hop_count, "answer_aliases": [str(x) for x in raw.get("answer_aliases", [])], "passages": passages, "supporting": supporting, "supporting_indices": supporting_indices, "source": "StonyBrookNLP/musique:musique_ans_v1.0_dev (HF pinned copy)"}


def wiki2_rows(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    for raw in data:
        passages = [(str(p[0]), " ".join(str(sentence) for sentence in p[1]), [str(sentence) for sentence in p[1]]) for p in raw["context"]]
        supporting = [(str(item[0]), int(item[1])) for item in raw.get("supporting_facts", [])]
        yield {"id": str(raw["_id"]), "question": str(raw["question"]), "answer": str(raw["answer"]), "type": str(raw.get("type") or "unknown"), "hop_count": None, "answer_aliases": [], "passages": passages, "supporting": supporting, "source": "Alab-NII/2wikimultihop:data/dev.json"}


def map_gold(record: dict, corpus_by_key: dict[tuple[str, str, str], dict]):
    docs = []
    ambiguities = []
    for index, (title, text, sentences) in enumerate(record["passages"]):
        title = canonical_text(title)
        text = canonical_text(text)
        doc_id = passage_id(record["dataset"], title, text)
        key = (record["dataset"], norm_key(title), text)
        doc = corpus_by_key.setdefault(key, {"dataset": record["dataset"], "doc_id": doc_id, "title": title, "full_text": text, "text_sha256": sha256_bytes(text.encode("utf-8")), "original_source_ids": []})
        doc["original_source_ids"].append(f"{record['query_id']}:context:{index}")
        docs.append((title, text, sentences, doc_id, index))

    # MuSiQue annotates support at paragraph position.  Titles are not
    # unique, so resolving by title would create false ambiguities.
    if record["dataset"] == "musique":
        indices = record.get("supporting_indices", [])
        if not indices or any(index < 0 or index >= len(docs) for index in indices):
            record["gold_doc_ids"] = []
            record["gold_mapping_valid"] = False
            record["mapping_ambiguities"] = [{"reason": "support paragraph index outside context"}]
        else:
            record["gold_doc_ids"] = sorted({docs[index][3] for index in indices})
            record["gold_mapping_valid"] = len(record["gold_doc_ids"]) == len(indices)
            record["mapping_ambiguities"] = [] if record["gold_mapping_valid"] else [{"reason": "duplicate support paragraph identity"}]
        record["context_doc_ids"] = [item[3] for item in docs]
        return record

    gold = []
    for title, sentence_id in record["supporting"]:
        matches = [item for item in docs if norm_key(item[0]) == norm_key(title)]
        if sentence_id is not None:
            exact = [item for item in matches if 0 <= sentence_id < len(item[2])]
            if not exact:
                ambiguities.append({"title": title, "sentence_id": sentence_id, "match_count": len(matches), "reason": "support sentence index outside paragraph"})
                continue
            matches = exact
        unique_ids = {item[3] for item in matches}
        if len(unique_ids) == 1 and matches:
            matches = [matches[0]]
        if len(matches) != 1:
            ambiguities.append({"title": title, "sentence_id": sentence_id, "match_count": len(matches)})
            continue
        item = matches[0]
        if sentence_id is not None and canonical_text(item[2][sentence_id]) not in item[1]:
            ambiguities.append({"title": title, "sentence_id": sentence_id, "reason": "support sentence not in canonical paragraph"})
            continue
        gold.append(item[3])
    record["gold_doc_ids"] = sorted(set(gold))
    record["gold_mapping_valid"] = bool(record["gold_doc_ids"]) and not ambiguities
    record["mapping_ambiguities"] = ambiguities
    record["context_doc_ids"] = [item[3] for item in docs]
    return record


def deterministic_partition(records: list[dict], seed: int, cal_n: int, test_n: int):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for record in records:
        stratum = record["query_type"] or "unknown"
        if record["dataset"] == "musique":
            stratum = f"hop_{record.get('hop_count') or 'unknown'}"
        groups[stratum].append(record)
    for values in groups.values():
        values.sort(key=lambda record: record["query_id"])
        rng.shuffle(values)
    total = cal_n + test_n
    ordered = []
    while groups and len(ordered) < total:
        for name in sorted(groups):
            if groups[name]:
                ordered.append(groups[name].pop())
                if len(ordered) == total:
                    break
            else:
                del groups[name]
                break
    if len(ordered) != total:
        raise ValueError(f"insufficient records for {cal_n}+{test_n}: {len(ordered)}")
    for index, record in enumerate(ordered):
        record["split"] = "cal" if index < cal_n else "test"
        record["sample_seed"] = seed
        record["sample_stratum"] = record["query_type"] if record["dataset"] != "musique" else f"hop_{record.get('hop_count') or 'unknown'}"
    return ordered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/study.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    raw = ROOT / "data/raw/vendor"
    source_records = {
        "hotpot": list(hotpot_rows(raw / "hotpot/hotpot_dev_distractor_v1.parquet")),
        "musique": list(musique_rows(raw / "musique/musique_ans_v1.0_dev.jsonl")),
        "wiki2": list(wiki2_rows(raw / "2wiki/data/dev.json")),
    }
    all_records = []
    corpus_by_key = {}
    for dataset, records in source_records.items():
        for item in records:
            item["dataset"] = dataset
            item["query_id"] = f"{dataset}:{item['id']}"
            item["query_type"] = item.pop("type")
            map_gold(item, corpus_by_key)
            all_records.append(item)

    selected = []
    split_manifest = []
    for dataset, records in source_records.items():
        for item in records:
            item["dataset"] = dataset
            item["query_id"] = f"{dataset}:{item['id']}"
            item["query_type"] = item.pop("type") if "type" in item else item.get("query_type", "unknown")
        valid_records = [item for item in records if item["gold_mapping_valid"]]
        part = deterministic_partition(valid_records, config["seed"], config["datasets"][dataset]["cal"], config["datasets"][dataset]["test"])
        selected.extend(part)
        split_manifest.extend({"query_id": item["query_id"], "dataset": dataset, "split": item["split"], "query_type": item["query_type"], "hop_count": item.get("hop_count"), "gold_doc_ids": item["gold_doc_ids"], "gold_mapping_valid": item["gold_mapping_valid"], "sample_seed": item["sample_seed"], "sample_stratum": item["sample_stratum"], "provenance": "REAL_DATA"} for item in part)

    # The debug quota uses independent training records where locally available.
    debug = []
    for dataset in ("hotpot", "musique", "wiki2"):
        if dataset == "hotpot" and (raw / "hotpot/hotpot_train.parquet").exists():
            candidates = list(hotpot_rows(raw / "hotpot/hotpot_train.parquet"))[:10]
        elif dataset == "musique" and (raw / "musique/musique_ans_v1.0_train.jsonl").exists():
            candidates = list(musique_rows(raw / "musique/musique_ans_v1.0_train.jsonl"))[:10]
        else:
            candidates = list(wiki2_rows(raw / "2wiki/data/train.json"))[:10] if dataset == "wiki2" else []
        if len(candidates) < 10:
            candidates = [item for item in source_records[dataset] if item["query_id"] not in {x["query_id"] for x in selected}][:10]
        for item in candidates[:10]:
            item["dataset"] = dataset
            item["query_id"] = f"{dataset}:debug:{item['id']}"
            item["query_type"] = item.pop("type") if "type" in item else item.get("query_type", "unknown")
            map_gold(item, corpus_by_key)
            item["split"] = "debug"
            item["provenance"] = "REAL_DATA_DEBUG"
            debug.append(item)

    corpus = sorted(corpus_by_key.values(), key=lambda item: item["doc_id"])
    selected_public = []
    for item in selected:
        selected_public.append({"dataset": item["dataset"], "query_id": item["query_id"], "split": item["split"], "question": item["question"], "answer": item["answer"], "answer_aliases": item["answer_aliases"], "query_type": item["query_type"], "hop_count": item.get("hop_count"), "gold_doc_ids": item["gold_doc_ids"], "context_doc_ids": item["context_doc_ids"], "gold_mapping_valid": item["gold_mapping_valid"], "mapping_ambiguities": item["mapping_ambiguities"], "provenance": "REAL_DATA"})

    audit = []
    for item in sorted(selected, key=lambda x: (x["dataset"], x["query_id"]))[:60]:
        audit.append({"dataset": item["dataset"], "query_id": item["query_id"], "question": item["question"], "answer": item["answer"], "gold_doc_ids": item["gold_doc_ids"], "gold_mapping_valid": item["gold_mapping_valid"], "mapping_ambiguities": item["mapping_ambiguities"], "context_doc_ids": item["context_doc_ids"], "source": item["source"]})

    write_jsonl(ROOT / "data/manifests/questions.jsonl", selected_public)
    write_jsonl(ROOT / "data/manifests/split_manifest.jsonl", split_manifest)
    write_jsonl(ROOT / "data/manifests/debug_manifest.jsonl", [{"dataset": x["dataset"], "query_id": x["query_id"], "question": x["question"], "split": "debug", "provenance": x["provenance"]} for x in debug])
    write_jsonl(ROOT / "data/corpus/passages.jsonl", corpus)
    write_jsonl(ROOT / "data/manifests/gold_mapping_audit.jsonl", audit)
    counts = {dataset: {split: sum(1 for x in selected_public if x["dataset"] == dataset and x["split"] == split) for split in ("cal", "test")} for dataset in source_records}
    invalid_source = [{"dataset": item["dataset"], "query_id": item["query_id"], "mapping_ambiguities": item["mapping_ambiguities"]} for records in source_records.values() for item in records if not item["gold_mapping_valid"]]
    mapping = {"selected": len(selected_public), "valid": sum(x["gold_mapping_valid"] for x in selected_public), "invalid": sum(not x["gold_mapping_valid"] for x in selected_public), "invalid_source_records": len(invalid_source), "audit_rows": len(audit), "debug_rows": len(debug), "counts": counts}
    write_json(ROOT / "data/manifests/m1_data_manifest.json", {"study_id": config["study_id"], "seed": config["seed"], "source_counts": {k: len(v) for k, v in source_records.items()}, "corpus_passages": len(corpus), "mapping": mapping, "invalid_source_records": invalid_source, "artifact_sha256": {"questions": sha256_bytes((ROOT / "data/manifests/questions.jsonl").read_bytes()), "split": sha256_bytes((ROOT / "data/manifests/split_manifest.jsonl").read_bytes()), "corpus": sha256_bytes((ROOT / "data/corpus/passages.jsonl").read_bytes()), "gold_audit": sha256_bytes((ROOT / "data/manifests/gold_mapping_audit.jsonl").read_bytes())}, "muSiQue_provenance": "official Google Drive quota-blocked; exact named file fetched from pinned HF copy dgslibisey/MuSiQue@763b65f844118a148e92bb88e7de5cb191b4c5dc", "created_utc": datetime.now(timezone.utc).isoformat()})
    write_json(ROOT / "results/audit/phase_state.json", {"current_milestone": "M1", "status": "DONE", "jev_calls": 0, "jev_execution_allowed": False, "data_manifest": str(ROOT / "data/manifests/m1_data_manifest.json")})
    print(json.dumps(mapping, indent=2, sort_keys=True))
    return 0 if mapping["selected"] == 750 and mapping["invalid"] == 0 and mapping["audit_rows"] == 60 else 2


if __name__ == "__main__":
    raise SystemExit(main())
