"""Tests for clone_and_assign module.

Uses mock CSV data so tests don't require the real
block_cd_distributions.csv.gz file.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from policyengine_us_data.calibration.clone_and_assign import (
    GeographyAssignment,
    load_global_block_distribution,
    assign_random_geography,
    assign_geography_within_state_county,
    double_geography_for_puf,
)

MOCK_BLOCKS = pd.DataFrame(
    {
        "cd_geoid": [101, 101, 101, 102, 102, 103, 103, 103, 103],
        "block_geoid": [
            "010010001001001",
            "010010001001002",
            "010010001001003",
            "020010001001001",
            "020010001001002",
            "360100001001001",
            "360100001001002",
            "360100001001003",
            "360100001001004",
        ],
        "probability": [
            0.4,
            0.3,
            0.3,
            0.6,
            0.4,
            0.25,
            0.25,
            0.25,
            0.25,
        ],
    }
)


@pytest.fixture(autouse=True)
def _clear_lru_cache():
    load_global_block_distribution.cache_clear()
    yield
    load_global_block_distribution.cache_clear()


def _mock_distribution():
    blocks = MOCK_BLOCKS["block_geoid"].values
    cds = MOCK_BLOCKS["cd_geoid"].astype(str).values
    states = np.array([int(b[:2]) for b in blocks])
    counties = np.array([int(b[:5]) for b in blocks])
    weights = MOCK_BLOCKS["probability"].values.astype(np.float64)
    return blocks, cds, states, counties, weights


class TestLoadGlobalBlockDistribution:
    def test_loads_distribution(self, tmp_path):
        csv_path = tmp_path / "block_cd_distributions.csv.gz"
        MOCK_BLOCKS.to_csv(csv_path, index=False, compression="gzip")
        with patch(
            "policyengine_us_data.calibration"
            ".clone_and_assign.STORAGE_FOLDER",
            tmp_path,
        ):
            blocks, cds, states, counties, weights = (
                load_global_block_distribution.__wrapped__()
            )
        assert len(blocks) == 9
        assert len(cds) == len(states) == len(counties) == len(weights) == 9

    def test_state_fips_extracted(self, tmp_path):
        csv_path = tmp_path / "block_cd_distributions.csv.gz"
        MOCK_BLOCKS.to_csv(csv_path, index=False, compression="gzip")
        with patch(
            "policyengine_us_data.calibration"
            ".clone_and_assign.STORAGE_FOLDER",
            tmp_path,
        ):
            _, _, states, _, _ = load_global_block_distribution.__wrapped__()
        assert states[0] == 1
        assert states[3] == 2
        assert states[5] == 36

    def test_county_fips_extracted(self, tmp_path):
        csv_path = tmp_path / "block_cd_distributions.csv.gz"
        MOCK_BLOCKS.to_csv(csv_path, index=False, compression="gzip")
        with patch(
            "policyengine_us_data.calibration"
            ".clone_and_assign.STORAGE_FOLDER",
            tmp_path,
        ):
            _, _, _, counties, _ = load_global_block_distribution.__wrapped__()
        assert counties[0] == 1001
        assert counties[3] == 2001
        assert counties[5] == 36010


class TestAssignRandomGeography:
    @patch(
        "policyengine_us_data.calibration.clone_and_assign"
        ".load_global_block_distribution"
    )
    def test_shape(self, mock_load):
        mock_load.return_value = _mock_distribution()
        r = assign_random_geography(n_records=10, n_clones=3, seed=42)
        assert len(r.block_geoid) == 30
        assert r.n_records == 10
        assert r.n_clones == 3

    @patch(
        "policyengine_us_data.calibration.clone_and_assign"
        ".load_global_block_distribution"
    )
    def test_deterministic(self, mock_load):
        mock_load.return_value = _mock_distribution()
        r1 = assign_random_geography(n_records=10, n_clones=3, seed=99)
        r2 = assign_random_geography(n_records=10, n_clones=3, seed=99)
        np.testing.assert_array_equal(r1.block_geoid, r2.block_geoid)

    @patch(
        "policyengine_us_data.calibration.clone_and_assign"
        ".load_global_block_distribution"
    )
    def test_different_seeds_differ(self, mock_load):
        mock_load.return_value = _mock_distribution()
        r1 = assign_random_geography(n_records=100, n_clones=3, seed=1)
        r2 = assign_random_geography(n_records=100, n_clones=3, seed=2)
        assert not np.array_equal(r1.block_geoid, r2.block_geoid)

    @patch(
        "policyengine_us_data.calibration.clone_and_assign"
        ".load_global_block_distribution"
    )
    def test_state_from_block(self, mock_load):
        mock_load.return_value = _mock_distribution()
        r = assign_random_geography(n_records=20, n_clones=5, seed=42)
        for i in range(len(r.block_geoid)):
            expected = int(r.block_geoid[i][:2])
            assert r.state_fips[i] == expected

    def test_missing_file_raises(self, tmp_path):
        fake = tmp_path / "nonexistent"
        fake.mkdir()
        with patch(
            "policyengine_us_data.calibration"
            ".clone_and_assign.STORAGE_FOLDER",
            fake,
        ):
            with pytest.raises(FileNotFoundError):
                load_global_block_distribution.__wrapped__()


class TestAssignGeographyWithinStateCounty:
    @patch(
        "policyengine_us_data.calibration.clone_and_assign"
        ".load_global_block_distribution"
    )
    def test_samples_within_county_when_available(self, mock_load):
        mock_load.return_value = _mock_distribution()
        r = assign_geography_within_state_county(
            state_fips=np.array([1, 2, 36]),
            county_fips=np.array([1, 1, 10]),
            seed=42,
        )
        assert all(
            block.startswith(prefix)
            for block, prefix in zip(
                r.block_geoid,
                ["01001", "02001", "36010"],
            )
        )

    @patch(
        "policyengine_us_data.calibration.clone_and_assign"
        ".load_global_block_distribution"
    )
    def test_falls_back_to_state_when_county_missing(self, mock_load):
        mock_load.return_value = _mock_distribution()
        r = assign_geography_within_state_county(
            state_fips=np.array([1, 2, 36]),
            county_fips=np.array([999, 0, None], dtype=object),
            seed=42,
        )
        assert [int(block[:2]) for block in r.block_geoid] == [1, 2, 36]

    @patch(
        "policyengine_us_data.calibration.clone_and_assign"
        ".load_global_block_distribution"
    )
    def test_respects_population_weights(self, mock_load):
        blocks = np.array(["010010001001001", "010010001001002"])
        mock_load.return_value = (
            blocks,
            np.array(["101", "101"]),
            np.array([1, 1]),
            np.array([1001, 1001]),
            np.array([99.0, 1.0]),
        )
        r = assign_geography_within_state_county(
            state_fips=np.ones(500, dtype=int),
            county_fips=np.ones(500, dtype=int),
            seed=42,
        )
        assert (r.block_geoid == blocks[0]).sum() > 450


class TestDoubleGeographyForPuf:
    def test_doubles_n_records(self):
        geo = GeographyAssignment(
            block_geoid=np.array(["010010001001001", "020010001001001"] * 3),
            cd_geoid=np.array(["101", "202"] * 3),
            state_fips=np.array([1, 2] * 3),
            county_fips=np.array([1001, 2001] * 3),
            n_records=2,
            n_clones=3,
        )
        r = double_geography_for_puf(geo)
        assert r.n_records == 4
        assert r.n_clones == 3
        assert len(r.block_geoid) == 12

    def test_puf_half_matches_cps_half(self):
        geo = GeographyAssignment(
            block_geoid=np.array(
                [
                    "010010001001001",
                    "020010001001001",
                    "360100001001001",
                    "060100001001001",
                    "480100001001001",
                    "120100001001001",
                ]
            ),
            cd_geoid=np.array(["101", "202", "1036", "653", "4831", "1227"]),
            state_fips=np.array([1, 2, 36, 6, 48, 12]),
            county_fips=np.array([1001, 2001, 36010, 6010, 48010, 12010]),
            n_records=3,
            n_clones=2,
        )
        r = double_geography_for_puf(geo)
        n_new = r.n_records

        for c in range(r.n_clones):
            start = c * n_new
            mid = start + n_new // 2
            end = start + n_new
            np.testing.assert_array_equal(
                r.state_fips[start:mid],
                r.state_fips[mid:end],
            )
