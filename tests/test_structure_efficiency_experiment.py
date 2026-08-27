import unittest

import numpy as np

from structure_efficiency_experiment import (
    _hypergraph_traversal,
    _pairwise_traversal,
    _relation_indices,
)


class StructureEfficiencyExperimentTests(unittest.TestCase):
    def test_pairwise_expansion_preserves_each_hyperedge_occurrence(self):
        active = np.asarray(
            [
                [True, False],
                [True, True],
                [True, False],
                [False, True],
            ]
        )
        vertex_ids, hyperedge_ids, pair_left, pair_right = _relation_indices(active)

        self.assertEqual(len(vertex_ids), 5)
        self.assertEqual(len(hyperedge_ids), 5)
        self.assertEqual(len(pair_left), 4)
        self.assertEqual(len(pair_right), 4)

    def test_both_traversals_consume_the_same_query_signals(self):
        active = np.asarray(
            [
                [True, False],
                [True, True],
                [True, False],
                [False, True],
            ]
        )
        vertex_ids, hyperedge_ids, pair_left, pair_right = _relation_indices(active)
        query_signals = np.asarray(
            [[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 2.0, 0.0]], dtype=np.float64
        )

        hypergraph = _hypergraph_traversal(
            query_signals,
            vertex_ids,
            hyperedge_ids,
            np.bincount(hyperedge_ids, minlength=2).astype(np.float64),
            4,
            2,
        )
        pairwise = _pairwise_traversal(query_signals, pair_left, pair_right, 4)

        self.assertTrue(np.isfinite(hypergraph))
        self.assertTrue(np.isfinite(pairwise))


if __name__ == "__main__":
    unittest.main()
