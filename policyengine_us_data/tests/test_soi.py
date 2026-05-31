import pandas as pd


def test_get_soi_upcasts_value_before_uprating(monkeypatch):
    from policyengine_us_data.utils import soi as soi_module

    source = pd.DataFrame(
        {
            "Year": [2022, 2022],
            "Variable": ["employment_income", "count"],
            "Value": pd.Series([100, 10], dtype="int64"),
        }
    )
    uprating = pd.DataFrame(
        {
            2022: {"employment_income": 1.0},
            2024: {"employment_income": 1.5},
        }
    )

    monkeypatch.setattr(soi_module.pd, "read_csv", lambda _: source.copy())
    monkeypatch.setattr(
        soi_module,
        "create_policyengine_uprating_factors_table",
        lambda: uprating,
    )

    result = soi_module.get_soi(2024)

    assert result["Value"].dtype == "float64"
    assert result.loc[result["Variable"] == "employment_income", "Value"].iloc[0] == 150.0
