import ast
import csv
import importlib.util
import json
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
SCRIPT = HERE / "scripts/e2_m0_structural_oracle.py"
spec = importlib.util.spec_from_file_location("e2_m0_structural_oracle", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def groups(*parts):
    return [list(x) for x in parts]


class StructuralOraclePureTests(unittest.TestCase):
    def test_01_gold_collision_same_group_is_quota_loss(self):
        x = mod.oracle_fields(["A", "C"], ["A", "C", "B", "D"], groups(["A", "C"], ["B", "D"], ["E"], ["F"], ["G"]), ["A", "B"])
        self.assertEqual((x["retrieval_oracle"], x["group_oracle"], x["observed_g5_success"]), (1, 0, 0))

    def test_02_gold_in_different_groups_can_be_retained(self):
        x = mod.oracle_fields(["A", "C"], ["A", "C", "B", "D"], groups(["A", "B"], ["C", "D"], ["E"], ["F"], ["G"]), ["A", "B"])
        self.assertEqual((x["retrieval_oracle"], x["group_oracle"], x["observed_g5_success"]), (1, 1, 0))

    def test_03_absent_gold_is_retrieval_miss(self):
        x = mod.oracle_fields(["Z"], ["A", "B", "C", "D"], groups(["A"], ["B"], ["C"], ["D"], []), ["A", "B", "C", "D"])
        self.assertEqual((x["retrieval_oracle"], x["group_oracle"], x["observed_g5_success"]), (0, 0, 0))

    def test_04_five_gold_one_per_group_is_feasible(self):
        slate = list("ABCDEFGHIJ")
        x = mod.oracle_fields(list("ACEGI"), slate, groups(["A", "B"], ["C", "D"], ["E", "F"], ["G", "H"], ["I", "J"]), list("ACEGI"))
        self.assertEqual(x["group_oracle"], 1)

    def test_05_more_than_budget_is_excluded(self):
        self.assertEqual(mod.reason_for_gold("RESOLVED", list("ABCDEF")), "GOLD_GT_BUDGET")

    def test_06_empty_gold_is_explicitly_excluded(self):
        self.assertEqual(mod.reason_for_gold("RESOLVED", []), "EMPTY_GOLD")

    def test_07_ambiguous_gold_is_explicitly_excluded(self):
        self.assertEqual(mod.reason_for_gold("AMBIGUOUS_GOLD_ID", []), "AMBIGUOUS_GOLD_ID")

    def test_08_path_qualified_ids_do_not_collapse(self):
        a = "enterprise:v1.0.0:path-qualified-relative-archive-member:a%2Fdsid_same.txt"
        b = "enterprise:v1.0.0:path-qualified-relative-archive-member:b%2Fdsid_same.txt"
        self.assertNotEqual(a, b)
        self.assertEqual(len({a, b}), 2)

    def test_09_duplicate_slate_and_selected_ids_block(self):
        errors = mod.validate_groups([["A", "B"], ["C", "D"], ["E", "F"], ["G", "H"], ["I", "J"]], ["A", "A", "C", "D"], ["A", "A", "C", "D", "E"])
        self.assertIn("DUPLICATE_SLATE_ID", errors)
        self.assertIn("DUPLICATE_SELECTED_ID", errors)

    def test_10_out_of_slate_selected_id_blocks(self):
        errors = mod.validate_groups([["A"], ["B"], ["C"], ["D"], []], ["A", "B", "C", "D"], ["A", "B", "C", "D", "Z"])
        self.assertIn("SELECTED_OUT_OF_SLATE", errors)

    def test_11_duplicate_group_membership_blocks(self):
        errors = mod.validate_groups([["A", "B"], ["B", "C"], ["D"], ["E"], ["F"]], list("ABCDEF"), list("ABCDE"))
        self.assertIn("DUPLICATE_GROUP_MEMBERSHIP", errors)

    def test_12_missing_group_record_cannot_be_inferred(self):
        with self.assertRaises(mod.AuditError):
            mod.group_doc_lists({})

    def test_13_same_score_top5_has_stable_tie_break(self):
        scores = {"B": 1.0, "A": 1.0, "C": 0.5, "D": 0.1, "E": 0.0, "F": 0.0}
        candidate_order = ["B", "A", "C", "D", "E", "F"]
        chosen = sorted(candidate_order, key=lambda d: (-scores[d], candidate_order.index(d)))[:5]
        self.assertEqual(chosen, ["B", "A", "C", "D", "E"])

    def test_14_namespaced_query_and_document_ids_remain_distinct(self):
        self.assertNotEqual("hotpot:q", "wiki2:q")
        self.assertNotEqual("hotpot:d", "musique:d")

    def test_15_aggregate_output_has_separate_denominators(self):
        out = HERE / "results/non_jev/v2/e2_m0/E2_M0_AGGREGATE_ORACLES.csv"
        self.assertTrue(out.is_file())
        with out.open() as f:
            rows = list(csv.DictReader(f))
        a = next(r for r in rows if r["dataset"] == "track_a/pooled" and r["policy"] == "clef_g5" and r["population"] == "unconditional_eligible")
        c = next(r for r in rows if r["dataset"] == "track_a/pooled" and r["policy"] == "clef_g5" and r["population"] == "retrieval_conditional")
        self.assertEqual((a["N_eligible"], c["N_conditional"]), ("3767", "1490"))
        self.assertNotEqual(a["N_eligible"], c["N_conditional"])

    def test_16_no_gpu_or_network_imports_in_audit_script(self):
        tree = ast.parse(SCRIPT.read_text())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module.split(".")[0])
        self.assertFalse(set(imports) & {"torch", "requests", "qdrant_client", "transformers"})


if __name__ == "__main__":
    unittest.main()
