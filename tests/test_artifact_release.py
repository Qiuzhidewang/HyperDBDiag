"""Check relocation, duplicate-case detection, and protection of frozen runs."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from ecsa_experiment import archived_source_member
from reproduce import validate_output
from verify_artifact import unique_rows, scores

class ArtifactReleaseTests(unittest.TestCase):
    def test_historical_absolute_source_is_resolved_after_relocation(self):
        members = ['src/experiment.py', 'tests/test_experiment.py']
        self.assertEqual(archived_source_member('/different/machine/project/src/experiment.py', members), 'src/experiment.py')
        with self.assertRaises(ValueError):
            archived_source_member('/different/machine/project/src/experiment.py', ['experiment.py', 'src/experiment.py'])
        with self.assertRaises(ValueError):
            archived_source_member('/different/machine/project/src/experiment.py', ['other/experiment.py'])

    def test_frozen_directories_and_nonempty_outputs_are_rejected(self):
        for output in [ROOT, ROOT.parent, ROOT / 'runs/new', ROOT / 'data/new', ROOT / 'src/new']:
            with self.assertRaises(ValueError):
                validate_output(output)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'fresh'
            self.assertEqual(validate_output(output), output.resolve())
            output.mkdir(); (output / 'result.json').write_text('{}')
            with self.assertRaises(ValueError):
                validate_output(output)

    def test_duplicate_cases_cannot_pass_an_aggregate_metric_check(self):
        rows = [{'case_id': 'a', 'expected_labels': ['x'], 'predicted_labels': ['x']}] * 2
        self.assertEqual(scores(rows)['Exact'], 1)
        with self.assertRaises(ValueError):
            unique_rows(rows, expected=2)

    def test_micro_f1_uses_root_counts_not_mean_case_f1(self):
        rows = [{'expected_labels': ['a'], 'predicted_labels': ['a']},
                {'expected_labels': ['a', 'b'], 'predicted_labels': ['b', 'c']}]
        result = scores(rows)
        self.assertEqual(result['Exact'], .5)
        self.assertAlmostEqual(result['F1'], 2 / 3)

if __name__ == '__main__':
    unittest.main()
