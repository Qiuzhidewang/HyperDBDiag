import json
import tempfile
import unittest
from pathlib import Path

from sql_semantic_evidence import load_frozen_case_observations


class SqlSemanticEvidenceTests(unittest.TestCase):
    def test_registered_semantic_evidence_aligns_with_frozen_cases(self):
        root = Path("data/dbmags_interaction_v10_metric_only")
        inputs = json.loads((root / "frozen_inputs.json").read_text(encoding="utf-8"))
        case_ids = [row["case_id"] for row in inputs["samples"]]
        observations = load_frozen_case_observations(
            root / "semantic_evidence.json", case_ids
        )
        self.assertEqual(len(observations), 660)
        self.assertEqual(sum(bool(row) for row in observations.values()), 396)
        summaries = " ".join(
            item.summary() for rows in observations.values() for item in rows
        ).lower()
        self.assertNotIn("select ", summaries)
        self.assertNotIn("customer", summaries)
        self.assertNotIn("orders", summaries)

    def test_reader_rejects_case_inventory_mismatch(self):
        payload = {
            "protocol": "dbmags-frozen-anonymous-semantic-evidence-v1",
            "samples": [],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "semantic.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inventories differ"):
                load_frozen_case_observations(path, ("case-0001",))


if __name__ == "__main__":
    unittest.main()
