import json
import tempfile
import unittest
from pathlib import Path

from metric_frozen_dataset import load_frozen_metric_dataset
from metric_frozen_schema import FEATURE_SCHEMA_SHA256


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n")


def _copy_files(source, destination):
    for path in Path(source).iterdir():
        if path.is_file():
            (Path(destination) / path.name).write_bytes(path.read_bytes())


class FrozenMetricDatasetTests(unittest.TestCase):
    SOURCE = Path("data/dbmags_interaction_v10_metric_only")

    def test_registered_artifact_is_complete_and_uses_one_outer_protocol(self):
        dataset = load_frozen_metric_dataset(self.SOURCE)
        manifest = _read_json(self.SOURCE / "dataset_manifest.json")
        self.assertEqual(len(dataset.case_ids), 660)
        self.assertEqual(dataset.feature_count, 25)
        self.assertEqual(dataset.replicate_count, 6)
        self.assertEqual(manifest["feature_schema_sha256"], FEATURE_SCHEMA_SHA256)
        self.assertEqual(
            manifest["outer_evaluation"],
            {
                "name": "leave_one_replicate_index_out",
                "group_field": "block_index",
            },
        )
        self.assertEqual(
            set(manifest["frozen_file_sha256"]),
            {
                "frozen_inputs.json",
                "fold_manifest.json",
                "ground_truth.json",
                "provenance_case_map.json",
                "semantic_evidence.json",
            },
        )

    def test_reader_rejects_tampered_frozen_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _copy_files(self.SOURCE, root)
            path = root / "frozen_inputs.json"
            payload = _read_json(path)
            payload["samples"][0]["metric_time_features"][0] += 1.0
            _write_json(path, payload)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_frozen_metric_dataset(root)

    def test_reader_rejects_alternate_outer_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _copy_files(self.SOURCE, root)
            path = root / "dataset_manifest.json"
            manifest = _read_json(path)
            manifest["outer_evaluation"]["name"] = "leave_one_scenario_out"
            _write_json(path, manifest)
            with self.assertRaisesRegex(ValueError, "outer evaluation"):
                load_frozen_metric_dataset(root)


if __name__ == "__main__":
    unittest.main()
