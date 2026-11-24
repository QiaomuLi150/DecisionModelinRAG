#!/usr/bin/env python3
"""Small CPU-only release integrity test suite (stdlib only)."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReleaseIntegrity(unittest.TestCase):
    def test_frozen_cqadupstack_protocol_and_cohort_hashes(self):
        self.assertEqual(sha(ROOT / "protocols/cqadupstack_frozen_protocol.md"), "f15cb0f6001327bfe47e0b1d605f73f7beefc3c6a5fc195689cd5ca1652ce86e")
        self.assertEqual(sha(ROOT / "protocols/cqadupstack_latency_cohort_ids.jsonl"), "5d5f34e3b3a9cfd1b547744e1e4417c992e97b86a81edd94f8f175645652961c")

    def test_bright_frozen_slates_have_1384_unique_domain_queries(self):
        seen = set()
        with gzip.open(ROOT / "data/candidates/bright_top20_ids.jsonl.gz", "rt", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                self.assertEqual(len(row.get("candidate_ids", [])), 20)
                self.assertEqual(len(set(row["candidate_ids"])), 20)
                seen.add((row.get("domain", ""), row["query_id"]))
        self.assertEqual(len(seen), 1384)

    def test_cqadupstack_scored_rows_and_candidate_identity(self):
        with (ROOT / "data/scores/cqadupstack_perquery.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len({row["canonical_id"] for row in rows}), 13125)
        groups = {}
        for row in rows:
            groups.setdefault(row["policy"], set()).add(row["canonical_id"])
        self.assertTrue(groups)
        self.assertTrue(all(len(ids) == 13125 for ids in groups.values()))

    def test_transfer_population_and_failure_accounting(self):
        with (ROOT / "data/scores/transfer_perquery.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        counts = {dataset: len({row["canonical_id"] for row in rows if row["dataset"] == dataset}) for dataset in {r["dataset"] for r in rows}}
        self.assertEqual(counts.get("enterprise"), 500)
        self.assertEqual(counts.get("musique"), 2167)
        self.assertNotIn(2500, counts.values())

    def test_trace_manifest_projection_hashes(self):
        with (ROOT / "raw_traces/TRACE_MANIFEST.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertGreaterEqual(len(rows), 1)
        for row in rows:
            path = ROOT / row["release_relative_path"]
            self.assertTrue(path.is_file(), row["release_relative_path"])
            self.assertEqual(sha(path), row["release_sha256"], row["release_relative_path"])

    def test_source_crosswalk_release_hashes(self):
        with (ROOT / "data/SOURCE_CROSSWALK.csv").open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        self.assertGreaterEqual(len(rows), 1)
        actual = {}
        for row in rows:
            target = row["release_relative_path"]
            actual.setdefault(target, sha(ROOT / target))
            self.assertEqual(actual[target], row["release_sha256"], target)
            self.assertEqual(len(row["source_sha256"]), 64)

    def test_bright_source_query_metrics_not_merged_with_e2n(self):
        expected = json.loads((ROOT / "verify/expected_numbers.json").read_text(encoding="utf-8"))
        self.assertEqual(expected["arxiv_id"], "2610.34816")
        e2n = (ROOT / "results/01_model_screen/e2_n_limited_screen/e2_n_scope_report.md").read_text(encoding="utf-8")
        self.assertIn("E2-N Expanded Model Benchmark", e2n)

    def test_model_downloader_is_opt_in_and_requires_safe_pins(self):
        script = ROOT / "scripts/download_model.py"
        help_result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
        self.assertEqual(help_result.returncode, 0)
        missing_pin_out = ROOT / "download-test-jina"
        missing_pin = subprocess.run([sys.executable, str(script), "--model", "jina", "--output-dir", str(missing_pin_out)], capture_output=True, text=True)
        self.assertNotEqual(missing_pin.returncode, 0)
        self.assertIn("full 40-64 character", missing_pin.stderr)
        self.assertFalse(missing_pin_out.exists())
        inside_repo = ROOT / "download-test-inside"
        rejected = subprocess.run([sys.executable, str(script), "--model", "pplx", "--output-dir", str(inside_repo)], capture_output=True, text=True)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("outside the public repository", rejected.stderr)
        self.assertFalse(inside_repo.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
