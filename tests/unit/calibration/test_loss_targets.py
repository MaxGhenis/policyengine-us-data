import numpy as np
import pandas as pd
import pytest

from policyengine_us_data.utils.loss import (
    AGE_BUCKETED_HEALTH_TARGETS,
    HARD_CODED_TOTALS,
    _add_real_estate_tax_targets,
    _get_aca_national_targets,
    _add_ctc_targets,
    _get_medicaid_national_targets,
    _load_aca_spending_and_enrollment_targets,
    _load_medicaid_enrollment_targets,
)


def test_aca_targets_roll_forward_to_2025():
    targets, data_year = _load_aca_spending_and_enrollment_targets(2025)

    assert data_year == 2025
    assert len(targets) == 51
    assert int(targets["enrollment"].sum()) == 21_822_894


def test_aca_targets_use_latest_available_year():
    _, data_year = _load_aca_spending_and_enrollment_targets(2026)
    assert data_year == 2025


def test_aca_targets_fall_back_to_earliest_available_year():
    _, data_year = _load_aca_spending_and_enrollment_targets(2023)
    assert data_year == 2024


def test_aca_national_targets_annualize_2025_state_file():
    spending, enrollment, data_year = _get_aca_national_targets(2025)

    assert data_year == 2025
    assert enrollment == 21_822_894
    assert spending == pytest.approx(143_951_057_388.72)


def test_medicaid_targets_roll_forward_to_2025():
    targets, data_year = _load_medicaid_enrollment_targets(2025)

    assert data_year == 2025
    assert len(targets) == 51
    assert int(targets["enrollment"].sum()) == 69_185_225


def test_medicaid_targets_fall_back_to_earliest_available_year():
    _, data_year = _load_medicaid_enrollment_targets(2023)
    assert data_year == 2024


def test_medicaid_national_targets_use_2025_values():
    spending, enrollment, data_year = _get_medicaid_national_targets(2025)

    assert data_year == 2025
    assert enrollment == 69_185_225
    assert spending == pytest.approx(1_000_645_800_000.0001)


def test_hardcoded_totals_drop_weak_survey_preservation_targets():
    removed_targets = {
        "alimony_income",
        "alimony_expense",
        "child_support_expense",
        "child_support_received",
        "health_insurance_premiums_without_medicare_part_b",
        "other_medical_expenses",
        "over_the_counter_health_expenses",
        "rent",
        "spm_unit_capped_housing_subsidy",
        "spm_unit_capped_work_childcare_expenses",
    }
    assert removed_targets.isdisjoint(HARD_CODED_TOTALS)


def test_age_bucketed_health_targets_keep_only_medicare_part_b():
    assert AGE_BUCKETED_HEALTH_TARGETS == ("medicare_part_b_premiums",)


class _FakeRealEstateTaxSimulation:
    def calculate(self, variable, map_to=None, period=None):
        values = {
            ("real_estate_taxes", None): [100.0, 0.0, 50.0, 0.0],
            ("tax_unit_is_filer", None): [1.0, 1.0],
            ("tax_unit_itemizes", None): [1.0, 0.0],
            ("state_code", "household"): ["CA", "NY"],
        }
        key = (variable, map_to)
        if key not in values:
            raise AssertionError(f"Unexpected calculate call {key!r}")
        return _FakeArrayResult(values[key])

    def map_result(self, values, source_entity, target_entity, how=None):
        arr = np.asarray(values, dtype=np.float32)
        if (source_entity, target_entity) == ("person", "tax_unit"):
            return np.array([arr[:2].sum(), arr[2:].sum()], dtype=np.float32)
        if (source_entity, target_entity) == ("tax_unit", "household"):
            return arr.astype(np.float32)
        raise AssertionError(
            f"Unexpected map_result call {(source_entity, target_entity, how)!r}"
        )


def test_add_real_estate_tax_targets(monkeypatch):
    monkeypatch.setattr(
        "policyengine_us_data.utils.loss.get_national_geography_soi_target",
        lambda variable, year: {"amount": 123_000.0, "count": 17.0},
    )
    monkeypatch.setattr(
        "policyengine_us_data.utils.loss.get_state_geography_soi_targets",
        lambda variable, year: [
            {"state_code": "CA", "amount": 100_000.0, "count": 10.0},
            {"state_code": "NY", "amount": 50_000.0, "count": 5.0},
        ],
    )

    targets, loss_matrix = _add_real_estate_tax_targets(
        pd.DataFrame(),
        [],
        _FakeRealEstateTaxSimulation(),
        2024,
    )

    assert targets == [123_000.0, 17.0, 100_000.0, 10.0, 50_000.0, 5.0]
    np.testing.assert_array_equal(
        loss_matrix["nation/irs/real_estate_taxes"],
        np.array([100.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        loss_matrix["nation/irs/real_estate_taxes_count"],
        np.array([1.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        loss_matrix["state/irs/real_estate_taxes/CA"],
        np.array([100.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        loss_matrix["state/irs/real_estate_taxes/NY"],
        np.array([0.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        loss_matrix["state/irs/real_estate_taxes_count/CA"],
        np.array([1.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        loss_matrix["state/irs/real_estate_taxes_count/NY"],
        np.array([0.0, 0.0], dtype=np.float32),
    )


class _FakeArrayResult:
    def __init__(self, values):
        self.values = np.asarray(values)


class _FakeSimulation:
    def __init__(self):
        self.calculate_calls = []
        self.map_result_calls = []

    def calculate(self, variable, map_to=None, period=None):
        self.calculate_calls.append((variable, map_to, period))
        values = {
            "refundable_ctc": [100.0, 0.0, 50.0],
            "non_refundable_ctc": [80.0, 10.0, 0.0],
        }
        if variable not in values:
            raise AssertionError(f"Unexpected variable {variable!r}")
        if map_to == "household":
            return _FakeArrayResult(values[variable])
        if map_to is None:
            return _FakeArrayResult(values[variable])
        raise AssertionError(f"Unexpected map_to {map_to!r}")

    def map_result(self, values, source_entity, target_entity, how=None):
        self.map_result_calls.append((source_entity, target_entity, how))
        assert source_entity == "tax_unit"
        assert target_entity == "household"
        return np.asarray(values, dtype=np.float32)


def test_add_ctc_targets(monkeypatch):
    monkeypatch.setattr(
        "policyengine_us_data.utils.loss.get_national_geography_soi_target",
        lambda variable, year: {
            "refundable_ctc": {"amount": 33_000.0, "count": 17.0},
            "non_refundable_ctc": {"amount": 81_000.0, "count": 37.0},
        }[variable],
    )
    sim = _FakeSimulation()

    targets, loss_matrix = _add_ctc_targets(
        pd.DataFrame(),
        [],
        sim,
        2024,
    )

    assert targets == [33_000.0, 17.0, 81_000.0, 37.0]
    np.testing.assert_array_equal(
        loss_matrix["nation/irs/refundable_ctc"],
        np.array([100.0, 0.0, 50.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        loss_matrix["nation/irs/refundable_ctc_count"],
        np.array([1.0, 0.0, 1.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        loss_matrix["nation/irs/non_refundable_ctc"],
        np.array([80.0, 10.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        loss_matrix["nation/irs/non_refundable_ctc_count"],
        np.array([1.0, 1.0, 0.0], dtype=np.float32),
    )
