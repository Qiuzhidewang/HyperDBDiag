import unittest

import numpy as np

from hypergraph_core import (
    BinaryMetricGraphResidualEncoder,
    EPDGPathPrior,
    HypergraphDiffusionResidualEncoder,
    LightweightHDiffusionClassifier,
    OrdinaryBinaryGraphClassifier,
)


class HypergraphCoreTests(unittest.TestCase):
    def setUp(self):
        self.labels = ["a", "b"]
        self.label_rows = [["a"], ["b"], ["a", "b"], ["a"], ["b"], ["a", "b"]]
        self.groups = [0, 1, 0, 1, 0, 1]
        self.vectors = np.asarray(
            [
                [2.0, 0.0, 0.2, -1.0],
                [0.0, -2.0, -0.2, 1.0],
                [2.0, -2.0, 1.0, 1.0],
                [1.8, 0.1, 0.3, -0.8],
                [-0.1, -1.8, -0.3, 0.8],
                [1.7, -1.7, 0.9, -0.9],
            ],
            dtype=np.float64,
        )

    def test_h_uses_nonnegative_signed_atoms_and_keeps_queries_out_of_h(self):
        encoder = HypergraphDiffusionResidualEncoder(activation_threshold=0.5).fit(
            self.vectors
        )
        propagation_before = encoder.propagation.copy()
        residual = encoder.transform(self.vectors[:2])

        self.assertEqual(encoder.reference_h_shape, (8, 6))
        self.assertEqual(residual.shape, (2, 16))
        self.assertTrue(np.all(np.isfinite(residual)))
        self.assertTrue(np.array_equal(encoder.propagation, propagation_before))
        self.assertEqual(encoder.metadata()["vertex_schema"], "fixed_signed_metric_atoms")
        self.assertEqual(encoder.metadata()["signed_vertex_count"], 8)
        self.assertEqual(encoder.metadata()["training_hyperedge_count"], 6)
        self.assertGreater(encoder.metadata()["nonzero_incidence_count"], 0)
        self.assertGreaterEqual(
            encoder.metadata()["equivalent_pairwise_occurrence_count"],
            encoder.metadata()["unique_pair_count_after_clique_deduplication"],
        )
        self.assertEqual(
            encoder.metadata()["dynamic_hyperedge_aggregation"],
            "cosine_topk_over_frozen_training_hyperedges",
        )
        self.assertFalse(encoder.metadata()["test_query_in_propagation"])
        self.assertFalse(encoder.metadata()["test_query_changes_training_hyperedges"])

    def test_h_model_fuses_raw_and_residual_candidate_decoders(self):
        model = LightweightHDiffusionClassifier(
            activation_threshold=0.5,
            seed=7,
            decoder_criterion="entropy",
            decoder_n_estimators=11,
        ).fit(self.vectors, self.label_rows, self.labels, self.groups)

        self.assertEqual(model.training_metadata["base_feature_count"], 4)
        self.assertEqual(model.training_metadata["h_diffusion_feature_count"], 16)
        self.assertEqual(model.training_metadata["combined_feature_count"], 20)
        self.assertEqual(model.training_metadata["decoder_criterion"], "entropy")
        self.assertEqual(model.training_metadata["decoder_n_estimators"], 11)
        self.assertEqual(
            model.training_metadata["decoder_fusion"],
            "group_oof_adaptive_probability_fusion",
        )
        branches = model.predict_branch_probabilities(self.vectors[:2])
        self.assertEqual({"raw", "h_residual", "epdg_prior", "fused"}, set(branches))
        np.testing.assert_allclose(
            branches["fused"],
            0.5 * (branches["raw"] + branches["h_residual"]),
        )
        probabilities = model.predict_proba(self.vectors[:2])
        np.testing.assert_allclose(probabilities, branches["fused"])
        self.assertEqual(probabilities.shape, (2, 3))
        self.assertTrue(np.allclose(np.sum(probabilities, axis=1), 1.0))

    def test_epdg_prior_uses_only_registered_available_metric_paths(self):
        prior = EPDGPathPrior(
            ("m0", "m1"),
            {"a": {"m0": 1.0}, "b": {"sql_template": 5.0}},
        ).fit(self.vectors[:, :2], ("a", "b"))
        self.assertEqual(prior.active_edge_count, 1)
        probabilities = prior.candidate_probabilities(
            np.asarray([[3.0, 0.0], [0.0, 3.0]]), ((0,), (1,))
        )
        self.assertEqual(probabilities.shape, (2, 2))
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0))
        self.assertEqual(prior.metadata()["unavailable_path_types"], [
            "sql_template", "execution_plan", "execution_operator"
        ])

    def test_epdg_prior_learns_stable_signed_root_metric_paths(self):
        groups = (0, 0, 0, 1, 1, 1)
        labels = (("a",), ("b",), ("a", "b"), ("a",), ("b",), ("a", "b"))
        prior = EPDGPathPrior(("m0", "m1"), minimum_path_strength=0.1).fit(
            self.vectors[:, :2], ("a", "b"), labels, groups
        )
        self.assertGreater(prior.learned_edge_count, 0)
        self.assertGreater(prior.root_scores(np.asarray([[3.0, 0.0]]))[0, 0], 0.0)

    def test_epdg_h_model_uses_registered_paths_in_single_group_nested_fold(self):
        model = LightweightHDiffusionClassifier(
            activation_threshold=0.5,
            seed=7,
            decoder_n_estimators=11,
            epdg_path_edges={"a": {"m0": 1.0}, "b": {"m1": 1.0}},
            epdg_feature_names=("m0", "m1", "m2", "m3"),
            enable_epdg=True,
        ).fit(self.vectors, self.label_rows, self.labels, self.groups)
        self.assertEqual(model.epdg_prior.registered_edge_count, 2)
        self.assertEqual(model.predict_proba(self.vectors[:2]).shape, (2, 3))

    def test_nested_h_can_freeze_outer_training_fusion_weight(self):
        model = LightweightHDiffusionClassifier(
            activation_threshold=0.5,
            seed=7,
            decoder_n_estimators=11,
            adaptive_fusion=False,
            fixed_fusion_weight=0.8,
        ).fit(self.vectors, self.label_rows, self.labels, self.groups)
        self.assertEqual(model.training_metadata["fusion_selection"]["status"], "fixed_outer_training_weight")
        self.assertEqual(model.fusion_weight, 0.8)
        branches = model.predict_branch_probabilities(self.vectors[:2])
        np.testing.assert_allclose(
            branches["fused"], 0.8 * branches["raw"] + 0.2 * branches["h_residual"]
        )

    def test_pairwise_graph_is_training_fold_only(self):
        encoder = BinaryMetricGraphResidualEncoder(neighbor_count=2).fit(self.vectors)
        propagation_before = encoder.propagation.copy()
        residual = encoder.transform(self.vectors[:2])
        self.assertEqual(residual.shape, (2, 4))
        self.assertTrue(np.array_equal(encoder.propagation, propagation_before))
        self.assertEqual(encoder.metadata()["vertex_count"], 4)
        self.assertFalse(encoder.metadata()["test_query_in_graph_construction"])

    def test_ordinary_graph_has_no_candidate_decoder_state(self):
        model = OrdinaryBinaryGraphClassifier(seed=7, n_estimators=11).fit(
            self.vectors, self.label_rows, self.labels, self.groups
        )
        self.assertEqual(model.training_metadata["candidate_inventory"], "none")
        self.assertEqual(
            model.training_metadata["set_decoder"],
            "none; independent root probabilities plus threshold",
        )
        self.assertTrue(
            model.training_metadata["encoder"]["pairwise_relation"].startswith(
                "absolute_pearson"
            )
        )
        self.assertFalse(hasattr(model, "candidates"))
        self.assertEqual(set(model.predict(self.vectors[:2], self.labels)[0]), {"a"})


if __name__ == "__main__":
    unittest.main()
