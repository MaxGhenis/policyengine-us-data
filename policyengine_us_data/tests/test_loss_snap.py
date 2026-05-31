from pathlib import Path

import numpy as np
import pandas as pd

from policyengine_us_data.datasets.cps import cps as cps_module
from policyengine_us_data.utils import loss


class _FakeResult:
    def __init__(self, values):
        self.values = np.asarray(values)


class _FakeSim:
    def __init__(self):
        self.calculate_calls: list[tuple[str, str | None]] = []

    def calculate(self, variable, map_to=None):
        self.calculate_calls.append((variable, map_to))
        if variable == "snap":
            return _FakeResult([100.0, 0.0])
        if variable == "state_code":
            return _FakeResult(["CA", "NY"])
        raise AssertionError(f"Unexpected variable: {variable}")

    def map_result(self, values, from_entity, to_entity, how=None):
        assert from_entity == "person"
        assert to_entity == "household"
        assert how == "value_from_first_person"
        return np.asarray(values)


class _FakeTaxBenefitSystem:
    variables = {"age": object(), "ssi": object()}


class _FakeSSISim:
    tax_benefit_system = _FakeTaxBenefitSystem()

    def calculate(self, variable, map_to=None):
        assert map_to == "person"
        if variable == "age":
            return _FakeResult([10, 40, 70, 70])
        if variable == "ssi":
            return _FakeResult([100, 100, 100, 0])
        raise AssertionError(f"Unexpected variable: {variable}")

    def map_result(self, values, from_entity, to_entity, how=None):
        assert from_entity == "person"
        assert to_entity == "household"
        assert how is None
        return np.asarray(values, dtype=float)


def test_snap_loss_metrics_use_calculated_snap(monkeypatch):
    monkeypatch.setattr(
        pd,
        "read_csv",
        lambda _: pd.DataFrame(
            {
                "GEO_ID": ["04000US06", "04000US36"],
                "Cost": [1.0, 1.0],
                "Households": [1.0, 1.0],
            }
        ),
    )
    sim = _FakeSim()

    matrix = loss._add_snap_metric_columns(pd.DataFrame(index=[0, 1]), sim)

    assert ("snap", "household") in sim.calculate_calls
    assert all(call[0] != "snap_reported" for call in sim.calculate_calls)
    assert matrix["US06/snap-cost"].tolist() == [100.0, 0.0]
    assert matrix["US06/snap-hhs"].tolist() == [1, 0]


def test_cps_builder_does_not_export_reported_program_amounts():
    source = Path(cps_module.__file__).read_text()

    assert 'cps["snap_reported"]' not in source
    assert 'cps["ssi_reported"]' not in source
    assert 'cps["tanf_reported"]' not in source


def test_legacy_loss_matrix_adds_ssi_count_targets():
    matrix = pd.DataFrame(index=range(4))
    targets = []

    loss._add_ssi_recipient_targets(matrix, targets, _FakeSSISim())

    assert matrix["nation/ssa/ssi_recipients/all"].tolist() == [
        1.0,
        1.0,
        1.0,
        0.0,
    ]
    assert matrix["nation/ssa/ssi_recipients/under_18"].tolist() == [
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    assert targets == [
        loss.SSI_RECIPIENT_TARGETS_2024["all"],
        loss.SSI_RECIPIENT_TARGETS_2024["under_18"],
        loss.SSI_RECIPIENT_TARGETS_2024["18_64"],
        loss.SSI_RECIPIENT_TARGETS_2024["65_plus"],
    ]
