"""Tests for unified_calibration module.

Focuses on rerandomize_takeup: verifies draws differ by
block and are reproducible within the same block.
"""

import numpy as np
import pytest

from policyengine_us_data.utils.randomness import seeded_rng


class TestRerandomizeTakeupSeeding:
    """Verify seeded_rng(var, salt=block) produces
    reproducible, block-dependent draws."""

    def test_same_block_same_draws(self):
        var = "takes_up_snap_if_eligible"
        block = "010010001001001"
        rng1 = seeded_rng(var, salt=block)
        rng2 = seeded_rng(var, salt=block)
        draws1 = rng1.random(100)
        draws2 = rng2.random(100)
        np.testing.assert_array_equal(draws1, draws2)

    def test_different_blocks_different_draws(self):
        var = "takes_up_snap_if_eligible"
        rng1 = seeded_rng(var, salt="010010001001001")
        rng2 = seeded_rng(var, salt="020010001001001")
        draws1 = rng1.random(100)
        draws2 = rng2.random(100)
        assert not np.array_equal(draws1, draws2)

    def test_different_vars_different_draws(self):
        block = "010010001001001"
        rng1 = seeded_rng("takes_up_snap_if_eligible", salt=block)
        rng2 = seeded_rng("takes_up_aca_if_eligible", salt=block)
        draws1 = rng1.random(100)
        draws2 = rng2.random(100)
        assert not np.array_equal(draws1, draws2)

    def test_draws_in_unit_interval(self):
        rng = seeded_rng(
            "takes_up_snap_if_eligible",
            salt="010010001001001",
        )
        draws = rng.random(10000)
        assert draws.min() >= 0.0
        assert draws.max() < 1.0

    def test_rate_comparison_produces_booleans(self):
        rng = seeded_rng(
            "takes_up_snap_if_eligible",
            salt="010010001001001",
        )
        draws = rng.random(10000)
        rate = 0.75
        result = draws < rate
        assert result.dtype == bool
        frac = result.mean()
        assert 0.70 < frac < 0.80


class TestSimpleTakeupConfig:
    """Verify the SIMPLE_TAKEUP_VARS config is well-formed."""

    def test_all_entries_have_required_keys(self):
        from policyengine_us_data.calibration.unified_calibration import (
            SIMPLE_TAKEUP_VARS,
        )

        for entry in SIMPLE_TAKEUP_VARS:
            assert "variable" in entry
            assert "entity" in entry
            assert "rate_key" in entry
            assert entry["entity"] in (
                "person",
                "tax_unit",
                "spm_unit",
            )

    def test_expected_count(self):
        from policyengine_us_data.calibration.unified_calibration import (
            SIMPLE_TAKEUP_VARS,
        )

        assert len(SIMPLE_TAKEUP_VARS) == 9


class TestFitL0Weights:
    def test_passes_target_groups_to_l0_model(self, monkeypatch):
        import sys
        import types

        import torch
        from scipy import sparse

        captured = {}

        class FakeSparseCalibrationWeights:
            def __init__(self, n_features, **kwargs):
                self.n_features = n_features

            def fit(self, **kwargs):
                captured.update(kwargs)

            def get_weights(self, deterministic=True):
                return torch.ones(self.n_features)

        l0_mod = types.ModuleType("l0")
        calibration_mod = types.ModuleType("l0.calibration")
        calibration_mod.SparseCalibrationWeights = FakeSparseCalibrationWeights
        monkeypatch.setitem(sys.modules, "l0", l0_mod)
        monkeypatch.setitem(sys.modules, "l0.calibration", calibration_mod)

        from policyengine_us_data.calibration.unified_calibration import (
            fit_l0_weights,
        )

        target_groups = np.array([0, 1])
        weights = fit_l0_weights(
            X_sparse=sparse.csr_matrix([[1.0, 0.0], [0.0, 1.0]]),
            targets=np.array([1.0, 2.0]),
            lambda_l0=1e-8,
            epochs=2,
            target_groups=target_groups,
        )

        np.testing.assert_array_equal(weights, np.ones(2))
        np.testing.assert_array_equal(captured["target_groups"], target_groups)
