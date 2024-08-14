import pytest


@pytest.mark.parametrize("year", [2022])
def test_policyengine_cps_generates(year: int):
    from policyengine_us_data.policyengine_cps import CPS_2022

    dataset_by_year = {
        2022: CPS_2022,
    }

    dataset_by_year[year](require=True)
