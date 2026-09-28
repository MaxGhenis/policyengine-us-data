"""The income-year 2025 (ASEC 2026) Census CPS loader.

The ASEC 2026 public-use person file drops SPM_BBSUBVAL, which the loader
required for every income year after 2020. These tests pin the URL map, the
year-keyed SPM column list, and an end-to-end ``generate()`` over a synthetic
ASEC 2026 archive with no network access.
"""

import io
import re
import zipfile

import pandas as pd
import pytest

from policyengine_us_data.datasets.cps import census_cps
from policyengine_us_data.datasets.cps.census_cps import (
    CPS_URL_BY_YEAR,
    PERSON_COLUMNS,
    SPM_BBSUBVAL_INCOME_YEARS,
    SPM_UNIT_COLUMNS,
    TAX_UNIT_COLUMNS,
    CensusCPS_2024,
    CensusCPS_2025,
    _resolve_person_usecols,
    spm_unit_columns_for_year,
)

URL_PATTERN = re.compile(
    r"^https://www2\.census\.gov/programs-surveys/cps/datasets/"
    r"(?P<survey>\d{4})/march/asecpub(?P<code>\d{2})csv\.zip$"
)


@pytest.mark.parametrize("income_year", sorted(CPS_URL_BY_YEAR))
def test_every_url_is_the_following_survey_year(income_year):
    match = URL_PATTERN.match(CPS_URL_BY_YEAR[income_year])
    assert match is not None
    assert int(match["survey"]) == income_year + 1
    assert match["code"] == str(income_year + 1)[-2:]


def test_income_year_2025_reads_the_asec_2026_archive():
    assert CPS_URL_BY_YEAR[2025] == (
        "https://www2.census.gov/programs-surveys/cps/datasets/2026/march/"
        "asecpub26csv.zip"
    )
    dataset = object.__new__(CensusCPS_2025)
    assert dataset.time_period == 2025
    assert dataset.name == "census_cps_2025"
    assert dataset.file_path.name == "census_cps_2025.h5"
    assert dataset._cps_download_url == CPS_URL_BY_YEAR[2025]


@pytest.mark.parametrize("income_year", range(1990, 2101))
def test_spm_columns_carry_bbsubval_exactly_in_income_years_2021_to_2024(
    income_year,
):
    columns = spm_unit_columns_for_year(income_year)
    assert ("SPM_BBSUBVAL" in columns) == (2021 <= income_year <= 2024)
    # Every other SPM column is always requested, in the original order.
    assert [c for c in SPM_UNIT_COLUMNS if c != "SPM_BBSUBVAL"] == [
        c for c in columns if c != "SPM_BBSUBVAL"
    ]
    assert columns == [c for c in SPM_UNIT_COLUMNS if c in columns]


def test_spm_bbsubval_window_matches_the_published_files():
    # ASEC 2022-2025 (income 2021-2024) carry SPM_BBSUBVAL; ASEC 2021 and 2026
    # do not. Earlier loader versions encoded the lower edge as `<= 2020`.
    assert list(SPM_BBSUBVAL_INCOME_YEARS) == [2021, 2022, 2023, 2024]
    assert "SPM_BBSUBVAL" in SPM_UNIT_COLUMNS


def test_spm_columns_for_a_year_do_not_alias_the_module_list():
    columns = spm_unit_columns_for_year(2024)
    columns.append("SENTINEL")
    assert "SENTINEL" not in SPM_UNIT_COLUMNS


def _asec_2026_person_header() -> list[str]:
    return [
        c
        for c in dict.fromkeys(PERSON_COLUMNS + SPM_UNIT_COLUMNS + TAX_UNIT_COLUMNS)
        if c != "SPM_BBSUBVAL"
    ]


def test_an_asec_2026_header_resolves_for_2025_and_refuses_for_2024():
    header = _asec_2026_person_header()
    usecols = _resolve_person_usecols(header, spm_unit_columns_for_year(2025))
    assert "SPM_BBSUBVAL" not in usecols
    assert "SPM_ID" in usecols
    with pytest.raises(KeyError, match="SPM_BBSUBVAL"):
        _resolve_person_usecols(header, spm_unit_columns_for_year(2024))


def _synthetic_asec_2026_archive() -> bytes:
    header = _asec_2026_person_header()
    rows = []
    for line, (age, wages, expr) in enumerate(
        ((40, 50_000, 1), (38, 20_000, 3), (10, 0, 5)), start=1
    ):
        row = dict.fromkeys(header, 0)
        row.update(
            PH_SEQ=1,
            PF_SEQ=1,
            P_SEQ=line,
            A_LINENO=line,
            A_AGE=age,
            A_SEX=1 + line % 2,
            A_MARITL=1 if age >= 18 else 7,
            A_SPOUSE=(2 if line == 1 else 1) if age >= 18 else 0,
            A_EXPRRP=expr,
            PEPAR1=1 if age < 18 else -1,
            PEPAR2=2 if age < 18 else -1,
            PECOHAB=-1,
            WSAL_VAL=wages,
            PTOTVAL=wages,
            A_FNLWGT=150_000,
            TAX_ID=7,
            SPM_ID=9,
            SPM_WEIGHT=150_000,
            SPM_NUMPER=3,
        )
        rows.append(row)
    person = pd.DataFrame(rows, columns=header)
    family = pd.DataFrame({"FH_SEQ": [1, 2], "FFPOS": [1, 1], "FTOTVAL": [70_000, 0]})
    household = pd.DataFrame(
        {"H_SEQ": [1, 2], "H_IDNUM": ["000000000000001", "000000000000002"]}
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("pppub26.csv", person.to_csv(index=False))
        archive.writestr("ffpub26.csv", family.to_csv(index=False))
        archive.writestr("hhpub26.csv", household.to_csv(index=False))
    return buffer.getvalue()


class _FakeResponse:
    status_code = 200

    def __init__(self, payload: bytes):
        self._payload = payload
        self.headers = {"content-length": str(len(payload))}

    def iter_content(self, chunk_size):
        for start in range(0, len(self._payload), chunk_size):
            yield self._payload[start : start + chunk_size]


def test_generate_writes_an_income_year_2025_store_without_spm_bbsubval(
    tmp_path, monkeypatch
):
    payload = _synthetic_asec_2026_archive()
    requested = []

    def fake_get(url, stream):
        requested.append(url)
        return _FakeResponse(payload)

    monkeypatch.setattr(census_cps.requests, "get", fake_get)
    dataset = object.__new__(CensusCPS_2025)
    dataset.file_path = tmp_path / "census_cps_2025.h5"

    dataset.generate()

    assert requested == [CPS_URL_BY_YEAR[2025]]
    with pd.HDFStore(dataset.file_path, mode="r") as store:
        assert sorted(store.keys()) == [
            "/family",
            "/household",
            "/person",
            "/spm_unit",
            "/tax_unit",
        ]
        person = store["person"]
        spm_unit = store["spm_unit"]
        household = store["household"]
        family = store["family"]
    assert "SPM_BBSUBVAL" not in person.columns
    assert list(spm_unit.columns) == spm_unit_columns_for_year(2025)
    assert spm_unit.index.tolist() == [9]
    assert person["CENSUS_TAX_ID"].tolist() == [7, 7, 7]
    # Only households and families that have persons are kept.
    assert household["H_SEQ"].tolist() == [1]
    assert family["FH_SEQ"].tolist() == [1]


def test_generate_for_2024_still_requires_spm_bbsubval(tmp_path, monkeypatch):
    payload = _synthetic_asec_2026_archive()
    with zipfile.ZipFile(io.BytesIO(payload)) as source:
        members = {name: source.read(name) for name in source.namelist()}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in members.items():
            archive.writestr(name.replace("26", "25"), data)
    monkeypatch.setattr(
        census_cps.requests,
        "get",
        lambda url, stream: _FakeResponse(buffer.getvalue()),
    )
    dataset = object.__new__(CensusCPS_2024)
    dataset.file_path = tmp_path / "census_cps_2024.h5"

    with pytest.raises(KeyError, match="SPM_BBSUBVAL"):
        dataset.generate()
