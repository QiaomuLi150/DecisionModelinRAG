import csv
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/non_jev/v2/e2_m0"


class FrozenMappingAuditTests(unittest.TestCase):
    def test_output_manifest_and_owner_gate_are_machine_readable(self):
        manifest = json.loads((OUT / "E2_M0_PROVENANCE_MANIFEST.json").read_text())
        gate = json.loads((OUT / "E2_M0_OWNER_GATE.json").read_text())
        self.assertEqual(manifest["execution_scope"]["gpu_calls"], 0)
        self.assertEqual(manifest["execution_scope"]["jev_calls"], 0)
        self.assertTrue(gate["stop"])

    def test_full_query_rows_and_unique_namespaces(self):
        with (OUT / "E2_M0_PER_QUERY_ORACLES.csv").open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 2 * (3767 + 500))
        self.assertEqual(len({(r["dataset"], r["query_id"], r["group_policy"]) for r in rows}), len(rows))

    def test_qst_0413_retained_but_not_eligible(self):
        with (OUT / "E2_M0_PER_QUERY_ORACLES.csv").open() as f:
            rows = [r for r in csv.DictReader(f) if r["query_id"] == "qst_0413"]
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["exclusion_reason"] for r in rows}, {"AMBIGUOUS_GOLD_ID"})
        self.assertEqual({r["eligible"] for r in rows}, {"0"})

    def test_all_core_gate_statuses_pass(self):
        gate = json.loads((OUT / "E2_M0_OWNER_GATE.json").read_text())
        for key, value in gate["gates"].items():
            if key == "M0_SCORE_ABLATION":
                self.assertEqual(value, "NOT_RUN_OPTIONAL")
            else:
                self.assertIn(value, {"PASS"})


if __name__ == "__main__":
    unittest.main()
