"""Tests for extended CPS QRF imputation functions.

Uses synthetic data to verify that:
1. Sequential QRF preserves covariance between imputed variables
2. CPS-only imputation uses PUF-imputed income (not CPS originals)
3. Variable lists don't overlap (no double-imputation)
4. Post-processing constraints enforce IRS caps and SS normalization
"""

import numpy as np
import pandas as pd
import pytest

from policyengine_us_data.datasets.cps.extended_cps import (
    IMPUTED_VARIABLES,
    OVERRIDDEN_IMPUTED_VARIABLES,
    CPS_ONLY_IMPUTED_VARIABLES,
    CPS_STAGE2_INCOME_PREDICTORS,
    DEMOGRAPHIC_PREDICTORS,
    STAGE1_EXTRA_PREDICTORS,
    apply_retirement_constraints,
    reconcile_ss_subcomponents,
)


class TestVariableListConsistency:
    """Variable lists should not overlap — no variable should be
    imputed by two different mechanisms."""

    def test_no_overlap_imputed_and_cps_only(self):
        overlap = set(IMPUTED_VARIABLES) & set(CPS_ONLY_IMPUTED_VARIABLES)
        assert (
            overlap == set()
        ), f"Variables in both IMPUTED and CPS_ONLY: {overlap}"

    def test_no_overlap_overridden_and_cps_only(self):
        overlap = set(OVERRIDDEN_IMPUTED_VARIABLES) & set(
            CPS_ONLY_IMPUTED_VARIABLES
        )
        assert (
            overlap == set()
        ), f"Variables in both OVERRIDDEN and CPS_ONLY: {overlap}"

    def test_overridden_is_subset_of_imputed(self):
        not_in_imputed = set(OVERRIDDEN_IMPUTED_VARIABLES) - set(
            IMPUTED_VARIABLES
        )
        assert (
            not_in_imputed == set()
        ), f"OVERRIDDEN vars not in IMPUTED: {not_in_imputed}"

    def test_stage2_income_predictors_in_imputed(self):
        """Stage-2 income predictors must come from stage-1 imputation."""
        for var in CPS_STAGE2_INCOME_PREDICTORS:
            assert var in IMPUTED_VARIABLES, (
                f"Stage-2 income predictor '{var}' not in "
                f"IMPUTED_VARIABLES — won't have PUF-imputed values"
            )

    def test_retirement_contributions_in_cps_only(self):
        """All 5 retirement contribution vars should be in CPS_ONLY."""
        expected = {
            "traditional_401k_contributions",
            "roth_401k_contributions",
            "traditional_ira_contributions",
            "roth_ira_contributions",
            "self_employed_pension_contributions",
        }
        missing = expected - set(CPS_ONLY_IMPUTED_VARIABLES)
        assert missing == set(), (
            f"Retirement contribution vars missing from "
            f"CPS_ONLY: {missing}"
        )

    def test_ss_subcomponents_in_cps_only(self):
        """All 4 SS sub-component vars should be in CPS_ONLY."""
        expected = {
            "social_security_retirement",
            "social_security_disability",
            "social_security_dependents",
            "social_security_survivors",
        }
        missing = expected - set(CPS_ONLY_IMPUTED_VARIABLES)
        assert missing == set(), (
            f"SS sub-component vars missing from CPS_ONLY: {missing}"
        )

    def test_nonexistent_vars_not_in_cps_only(self):
        """Variables that don't exist in policyengine-us should not be
        in CPS_ONLY_IMPUTED_VARIABLES."""
        should_not_exist = {
            "roth_ira_distributions",
            "regular_ira_distributions",
            "other_type_retirement_account_distributions",
        }
        present = should_not_exist & set(CPS_ONLY_IMPUTED_VARIABLES)
        assert present == set(), (
            f"Non-existent variables still in CPS_ONLY: {present}"
        )

    def test_pension_income_not_in_cps_only(self):
        """Pension income vars are handled by Stage 1 rename, not
        Stage 2 QRF."""
        should_not_be_here = {
            "taxable_private_pension_income",
            "tax_exempt_private_pension_income",
        }
        present = should_not_be_here & set(CPS_ONLY_IMPUTED_VARIABLES)
        assert present == set(), (
            f"Pension income vars should not be in CPS_ONLY: {present}"
        )


class TestRetirementConstraints:
    """Post-processing retirement constraints enforce IRS caps."""

    @pytest.fixture
    def sample_predictions(self):
        """QRF predictions with values that need constraining."""
        return pd.DataFrame(
            {
                "traditional_401k_contributions": [
                    25000, -500, 5000, 10000, 3000
                ],
                "roth_401k_contributions": [
                    30000, 2000, 0, 50000, 1000
                ],
                "traditional_ira_contributions": [
                    8000, -100, 3000, 15000, 500
                ],
                "roth_ira_contributions": [
                    10000, 1000, 0, 20000, 200
                ],
                "self_employed_pension_contributions": [
                    80000, -200, 5000, 0, 100000
                ],
            }
        )

    @pytest.fixture
    def sample_features(self):
        return pd.DataFrame(
            {
                "age": [55, 30, 45, 60, 25],
                "employment_income": [100000, 50000, 0, 80000, 60000],
                "self_employment_income": [0, 0, 20000, 50000, 200000],
            }
        )

    def test_non_negativity(self, sample_predictions, sample_features):
        result = apply_retirement_constraints(
            sample_predictions, sample_features, 2024
        )
        for var in result.columns:
            assert (result[var] >= 0).all(), (
                f"{var} has negative values after constraints"
            )

    def test_401k_capped_at_limit(
        self, sample_predictions, sample_features
    ):
        result = apply_retirement_constraints(
            sample_predictions, sample_features, 2024
        )
        from policyengine_us_data.utils.retirement_limits import (
            get_retirement_limits,
        )

        limits = get_retirement_limits(2024)
        age = sample_features["age"].values
        catch_up = age >= 50
        cap = limits["401k"] + catch_up * limits["401k_catch_up"]
        for var in [
            "traditional_401k_contributions",
            "roth_401k_contributions",
        ]:
            assert (result[var].values <= cap).all(), (
                f"{var} exceeds 401k cap"
            )

    def test_ira_capped_at_limit(
        self, sample_predictions, sample_features
    ):
        result = apply_retirement_constraints(
            sample_predictions, sample_features, 2024
        )
        from policyengine_us_data.utils.retirement_limits import (
            get_retirement_limits,
        )

        limits = get_retirement_limits(2024)
        age = sample_features["age"].values
        catch_up = age >= 50
        cap = limits["ira"] + catch_up * limits["ira_catch_up"]
        for var in [
            "traditional_ira_contributions",
            "roth_ira_contributions",
        ]:
            assert (result[var].values <= cap).all(), (
                f"{var} exceeds IRA cap"
            )

    def test_401k_zeroed_without_employment_income(
        self, sample_predictions, sample_features
    ):
        result = apply_retirement_constraints(
            sample_predictions, sample_features, 2024
        )
        no_emp = sample_features["employment_income"] == 0
        for var in [
            "traditional_401k_contributions",
            "roth_401k_contributions",
        ]:
            assert (result[var].values[no_emp] == 0).all(), (
                f"{var} should be zero without employment income"
            )

    def test_se_pension_capped(
        self, sample_predictions, sample_features
    ):
        result = apply_retirement_constraints(
            sample_predictions, sample_features, 2024
        )
        se_income = sample_features["self_employment_income"].values
        se_vals = result["self_employed_pension_contributions"].values
        # Should be <= 25% of SE income (approximately)
        rate_cap = se_income * 0.25
        assert (se_vals <= rate_cap + 1).all(), (
            "SE pension exceeds 25% of SE income"
        )

    def test_se_pension_zeroed_without_se_income(
        self, sample_predictions, sample_features
    ):
        result = apply_retirement_constraints(
            sample_predictions, sample_features, 2024
        )
        no_se = sample_features["self_employment_income"] == 0
        assert (
            result["self_employed_pension_contributions"].values[no_se] == 0
        ).all(), "SE pension should be zero without SE income"


class TestSSReconciliation:
    """Post-processing SS normalization ensures sub-components
    sum to the total social_security."""

    def test_subcomponents_sum_to_total(self):
        predictions = pd.DataFrame(
            {
                "social_security_retirement": [0.6, 0.0, 0.8, 0.3],
                "social_security_disability": [0.3, 0.0, 0.1, 0.5],
                "social_security_dependents": [0.05, 0.0, 0.05, 0.1],
                "social_security_survivors": [0.05, 0.0, 0.05, 0.1],
            }
        )
        total_ss = np.array([20000, 0, 15000, 10000])
        result = reconcile_ss_subcomponents(predictions, total_ss)
        sums = sum(result[col].values for col in result.columns)
        np.testing.assert_allclose(
            sums, total_ss, atol=0.01,
            err_msg="SS sub-components don't sum to total",
        )

    def test_zero_ss_zeroes_all_subcomponents(self):
        predictions = pd.DataFrame(
            {
                "social_security_retirement": [0.5, 0.7],
                "social_security_disability": [0.3, 0.2],
                "social_security_dependents": [0.1, 0.05],
                "social_security_survivors": [0.1, 0.05],
            }
        )
        total_ss = np.array([0, 0])
        result = reconcile_ss_subcomponents(predictions, total_ss)
        for col in result.columns:
            assert (result[col].values == 0).all(), (
                f"{col} should be zero when total SS is zero"
            )

    def test_shares_are_non_negative(self):
        predictions = pd.DataFrame(
            {
                "social_security_retirement": [-0.5, 0.8],
                "social_security_disability": [1.2, 0.2],
                "social_security_dependents": [0.1, 0.0],
                "social_security_survivors": [0.2, 0.0],
            }
        )
        total_ss = np.array([10000, 5000])
        result = reconcile_ss_subcomponents(predictions, total_ss)
        for col in result.columns:
            assert (result[col].values >= 0).all(), (
                f"{col} has negative values"
            )

    def test_single_component_gets_full_total(self):
        """When only one sub-component has a positive share,
        it should get the full total."""
        predictions = pd.DataFrame(
            {
                "social_security_retirement": [1.0],
                "social_security_disability": [0.0],
                "social_security_dependents": [0.0],
                "social_security_survivors": [0.0],
            }
        )
        total_ss = np.array([25000])
        result = reconcile_ss_subcomponents(predictions, total_ss)
        assert result["social_security_retirement"].values[0] == pytest.approx(
            25000, abs=0.01
        )


class TestSequentialQRF:
    """Verify that sequential QRF produces correlated outputs,
    unlike independent imputation."""

    @pytest.fixture
    def correlated_training_data(self):
        """Synthetic data where y1 and y2 are correlated through x."""
        rng = np.random.default_rng(42)
        n = 2000
        x = rng.normal(50, 15, n)
        # y1 strongly correlated with x
        y1 = 0.8 * x + rng.normal(0, 5, n)
        # y2 correlated with both x and y1
        y2 = 0.3 * x + 0.5 * y1 + rng.normal(0, 3, n)
        return pd.DataFrame({"x": x, "y1": y1, "y2": y2})

    def test_sequential_qrf_preserves_correlation(
        self, correlated_training_data
    ):
        """When y2 depends on y1, sequential QRF should produce
        y1-y2 correlation closer to training data than independent
        imputation would."""
        from microimpute.models.qrf import QRF

        df = correlated_training_data
        train = df.sample(1500, random_state=0)
        test_x = df.drop(train.index)[["x"]]

        # Sequential: y2 conditions on y1
        qrf = QRF(log_level="ERROR")
        fitted = qrf.fit(
            X_train=train,
            predictors=["x"],
            imputed_variables=["y1", "y2"],
            n_jobs=1,
        )
        result = fitted.predict(X_test=test_x)

        # The imputed y1 and y2 should be positively correlated
        corr = result["y1"].corr(result["y2"])
        assert corr > 0.5, (
            f"Sequential QRF y1-y2 correlation = {corr:.3f}, "
            f"expected > 0.5"
        )

    def test_single_call_vs_separate_calls_differ(
        self, correlated_training_data
    ):
        """Imputing y1 and y2 in a single sequential call should
        produce different y2 values than imputing them separately
        (the old batched approach)."""
        from microimpute.models.qrf import QRF

        df = correlated_training_data
        train = df.sample(1500, random_state=0)
        test_x = df.drop(train.index)[["x"]]

        # Sequential (single call)
        qrf_seq = QRF(log_level="ERROR")
        fitted_seq = qrf_seq.fit(
            X_train=train,
            predictors=["x"],
            imputed_variables=["y1", "y2"],
            n_jobs=1,
        )
        result_seq = fitted_seq.predict(X_test=test_x)

        # Independent (separate calls, like old batched approach)
        qrf_y1 = QRF(log_level="ERROR")
        fitted_y1 = qrf_y1.fit(
            X_train=train[["x", "y1"]],
            predictors=["x"],
            imputed_variables=["y1"],
            n_jobs=1,
        )
        result_y1 = fitted_y1.predict(X_test=test_x)

        qrf_y2 = QRF(log_level="ERROR")
        fitted_y2 = qrf_y2.fit(
            X_train=train[["x", "y2"]],
            predictors=["x"],
            imputed_variables=["y2"],
            n_jobs=1,
        )
        result_y2 = fitted_y2.predict(X_test=test_x)

        # The sequential y1-y2 correlation should be higher than
        # the independent one
        corr_seq = result_seq["y1"].corr(result_seq["y2"])
        corr_indep = result_y1["y1"].corr(result_y2["y2"])

        assert corr_seq > corr_indep, (
            f"Sequential corr ({corr_seq:.3f}) should exceed "
            f"independent corr ({corr_indep:.3f})"
        )
