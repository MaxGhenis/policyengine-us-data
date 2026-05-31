import logging
from typing import Type

import numpy as np
import pandas as pd
from policyengine_core.data import Dataset

from policyengine_us_data.datasets.cps.cps import *  # noqa: F403
from policyengine_us_data.datasets.puf import *  # noqa: F403
from policyengine_us_data.storage import STORAGE_FOLDER

logger = logging.getLogger(__name__)


def _calculate_spm_thresholds_from_assigned_geography(
    data: dict[str, dict[int, np.ndarray]],
    cd_geoid: np.ndarray,
    time_period: int,
) -> np.ndarray:
    from policyengine_us_data.datasets.cps.local_area_calibration.calibration_utils import (
        load_cd_geoadj_values,
    )
    from policyengine_us_data.utils.spm import (
        calculate_spm_thresholds_with_geoadj,
    )

    spm_unit_ids = data["spm_unit_id"][time_period]
    person_spm_unit_ids = data["person_spm_unit_id"][time_period]
    person_household_ids = data["person_household_id"][time_period]
    household_ids = data["household_id"][time_period]
    ages = data["age"][time_period]

    cd_geoid = np.asarray(cd_geoid).astype(str)
    cd_geoadj_values = load_cd_geoadj_values(sorted(set(cd_geoid)))
    household_geoadj = np.array(
        [cd_geoadj_values.get(cd, 1.0) for cd in cd_geoid],
        dtype=float,
    )
    geoadj_by_household = dict(zip(household_ids, household_geoadj))

    person_df = pd.DataFrame(
        {
            "spm_unit_id": person_spm_unit_ids,
            "household_id": person_household_ids,
            "is_adult": ages >= 18,
            "is_child": ages < 18,
        }
    )
    spm_df = person_df.groupby("spm_unit_id").agg(
        num_adults=("is_adult", "sum"),
        num_children=("is_child", "sum"),
        household_id=("household_id", "first"),
    )
    spm_df = spm_df.reindex(spm_unit_ids)

    tenure = data.get("spm_unit_tenure_type", {}).get(time_period)
    tenure_codes = np.full(len(spm_unit_ids), 3, dtype=int)
    if tenure is not None:
        tenure_values = np.asarray(tenure)
        if np.issubdtype(tenure_values.dtype, np.bytes_):
            tenure_values = np.char.decode(tenure_values, "utf-8")
        tenure_codes = (
            pd.Series(tenure_values)
            .map(
                {
                    "OWNER_WITH_MORTGAGE": 1,
                    "OWNER_WITHOUT_MORTGAGE": 2,
                    "RENTER": 3,
                }
            )
            .fillna(3)
            .astype(int)
            .values
        )

    geoadj = spm_df["household_id"].map(geoadj_by_household).fillna(1.0).values
    return calculate_spm_thresholds_with_geoadj(
        num_adults=spm_df["num_adults"].fillna(0).values,
        num_children=spm_df["num_children"].fillna(0).values,
        tenure_codes=tenure_codes,
        geoadj=geoadj,
        year=time_period,
    )


class ExtendedCPS(Dataset):
    cps: Type[CPS]
    puf: Type[PUF]
    data_format = Dataset.TIME_PERIOD_ARRAYS

    def generate(self):
        from policyengine_us import Microsimulation

        from policyengine_us_data.calibration.clone_and_assign import (
            assign_geography_within_state_county,
        )
        from policyengine_us_data.calibration.puf_impute import (
            puf_clone_dataset,
        )

        logger.info("Loading CPS dataset: %s", self.cps)
        cps_sim = Microsimulation(dataset=self.cps)
        data = cps_sim.dataset.load_dataset()
        del cps_sim

        data_dict = {}
        for var in data:
            data_dict[var] = {self.time_period: data[var][...]}

        state_fips = data_dict["state_fips"][self.time_period]
        county_fips = data_dict.get("county_fips", {}).get(self.time_period)
        geography = assign_geography_within_state_county(
            state_fips=state_fips,
            county_fips=county_fips,
            seed=42,
        )
        data_dict["spm_unit_spm_threshold"] = {
            self.time_period: _calculate_spm_thresholds_from_assigned_geography(
                data_dict,
                geography.cd_geoid,
                self.time_period,
            )
        }

        logger.info("PUF clone with dataset: %s", self.puf)
        new_data = puf_clone_dataset(
            data=data_dict,
            state_fips=geography.state_fips,
            block_geoid=geography.block_geoid,
            cd_geoid=geography.cd_geoid,
            county_fips=geography.county_fips,
            time_period=self.time_period,
            puf_dataset=self.puf,
            dataset_path=str(self.cps.file_path),
        )

        self.save_dataset(new_data)


class ExtendedCPS_2024(ExtendedCPS):
    cps = CPS_2024_Full
    puf = PUF_2024
    name = "extended_cps_2024"
    label = "Extended CPS (2024)"
    file_path = STORAGE_FOLDER / "extended_cps_2024.h5"
    time_period = 2024


if __name__ == "__main__":
    ExtendedCPS_2024().generate()
