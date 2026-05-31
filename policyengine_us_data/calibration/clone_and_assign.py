"""Clone CPS records and assign random geography."""

import logging
from functools import lru_cache
from dataclasses import dataclass

import numpy as np
import pandas as pd

from policyengine_us_data.storage import STORAGE_FOLDER

logger = logging.getLogger(__name__)


@dataclass
class GeographyAssignment:
    """Random geography assignment for cloned CPS records.

    All arrays have length n_records * n_clones.
    Index i corresponds to clone i // n_records,
    record i % n_records.
    """

    block_geoid: np.ndarray  # str array, 15-char block GEOIDs
    cd_geoid: np.ndarray  # str array of CD GEOIDs
    state_fips: np.ndarray  # int array of 2-digit state FIPS
    county_fips: np.ndarray  # int array of 5-digit county FIPS
    n_records: int
    n_clones: int


@lru_cache(maxsize=1)
def load_global_block_distribution():
    """Load block_cd_distributions.csv.gz and build
    global distribution.

    Returns:
        Tuple of (block_geoids, cd_geoids, state_fips, county_fips,
        weights) where each is a numpy array indexed by block row. Weights
        are block populations when available, otherwise the legacy
        probability column.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    csv_path = STORAGE_FOLDER / "block_cd_distributions.csv.gz"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. "
            "Run make_block_cd_distributions.py to generate."
        )

    df = pd.read_csv(csv_path, dtype={"block_geoid": str})

    block_geoids = df["block_geoid"].str.zfill(15).values
    cd_geoids = df["cd_geoid"].astype(str).values
    state_fips = np.array([int(b[:2]) for b in block_geoids])
    county_fips = np.array([int(b[:5]) for b in block_geoids])

    weight_col = "population" if "population" in df.columns else "probability"
    weights = df[weight_col].values.astype(np.float64)

    return block_geoids, cd_geoids, state_fips, county_fips, weights


def _group_indices(values: np.ndarray) -> dict[int, np.ndarray]:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    if len(sorted_values) == 0:
        return {}
    starts = np.r_[
        0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1
    ]
    ends = np.r_[starts[1:], len(sorted_values)]
    return {
        int(sorted_values[start]): order[start:end]
        for start, end in zip(starts, ends)
    }


def _normalise_county_fips(state_fips, county_fips) -> int | None:
    if county_fips is None or pd.isna(county_fips):
        return None
    try:
        county_int = int(county_fips)
        state_int = int(state_fips)
    except (TypeError, ValueError):
        return None
    if county_int <= 0:
        return None
    if county_int >= 1_000:
        return county_int
    return state_int * 1_000 + county_int


def _sample_index(
    rng: np.random.Generator,
    candidates: np.ndarray,
    weights: np.ndarray,
    probabilities_cache: dict[tuple[int, int, int], np.ndarray],
) -> int:
    cache_key = (len(candidates), int(candidates[0]), int(candidates[-1]))
    if cache_key not in probabilities_cache:
        candidate_weights = weights[candidates]
        total = candidate_weights.sum()
        probabilities_cache[cache_key] = (
            candidate_weights / total
            if total > 0
            else np.full(len(candidates), 1 / len(candidates))
        )
    return int(rng.choice(candidates, p=probabilities_cache[cache_key]))


def assign_random_geography(
    n_records: int,
    n_clones: int = 10,
    seed: int = 42,
) -> GeographyAssignment:
    """Assign random census block geography to cloned
    CPS records.

    Each of n_records * n_clones total records gets a
    random census block sampled from the global
    population-weighted distribution. State and CD are
    derived from the block GEOID.

    Args:
        n_records: Number of households in the base CPS
            dataset.
        n_clones: Number of clones (default 10).
        seed: Random seed for reproducibility.

    Returns:
        GeographyAssignment with arrays of length
        n_records * n_clones.
    """
    blocks, cds, states, counties, weights = load_global_block_distribution()

    n_total = n_records * n_clones
    rng = np.random.default_rng(seed)
    probs = weights / weights.sum()
    indices = rng.choice(len(blocks), size=n_total, p=probs)

    return GeographyAssignment(
        block_geoid=blocks[indices],
        cd_geoid=cds[indices],
        state_fips=states[indices],
        county_fips=counties[indices],
        n_records=n_records,
        n_clones=n_clones,
    )


def assign_geography_within_state_county(
    state_fips: np.ndarray,
    county_fips: np.ndarray | None = None,
    n_clones: int = 1,
    seed: int = 42,
) -> GeographyAssignment:
    """Assign blocks within each record's state, and county when available.

    Sampling is weighted by block population when the block distribution file
    includes a population column, otherwise by its legacy probability weights.
    """
    blocks, cds, states, counties, weights = load_global_block_distribution()
    state_groups = _group_indices(states)
    county_groups = _group_indices(counties)

    input_states = np.tile(np.asarray(state_fips, dtype=int), n_clones)
    if county_fips is None:
        input_counties = np.array([None] * len(input_states), dtype=object)
    else:
        input_counties = np.tile(
            np.asarray(county_fips, dtype=object), n_clones
        )

    rng = np.random.default_rng(seed)
    probabilities_cache: dict[tuple[int, int, int], np.ndarray] = {}
    selected = np.empty(len(input_states), dtype=int)

    global_candidates = np.arange(len(blocks))
    for i, (state, county) in enumerate(zip(input_states, input_counties)):
        county_key = _normalise_county_fips(state, county)
        candidates = (
            county_groups.get(county_key) if county_key is not None else None
        )
        if candidates is None or len(candidates) == 0:
            candidates = state_groups.get(int(state))
        if candidates is None or len(candidates) == 0:
            logger.warning(
                "No block distribution for state=%s county=%s; using national distribution",
                state,
                county,
            )
            candidates = global_candidates
        selected[i] = _sample_index(
            rng, candidates, weights, probabilities_cache
        )

    return GeographyAssignment(
        block_geoid=blocks[selected],
        cd_geoid=cds[selected],
        state_fips=states[selected],
        county_fips=counties[selected],
        n_records=len(state_fips),
        n_clones=n_clones,
    )


def double_geography_for_puf(
    geography: GeographyAssignment,
) -> GeographyAssignment:
    """Double geography arrays for PUF clone step.

    After PUF cloning doubles the base records, the geography
    assignment must also double: each record and its PUF copy
    share the same geographic assignment.

    The output has n_records = 2 * geography.n_records, with
    the first half being the CPS records and the second half
    being the PUF copies.

    Args:
        geography: Original geography assignment.

    Returns:
        New GeographyAssignment with doubled n_records.
    """
    n_old = geography.n_records
    n_new = n_old * 2
    n_clones = geography.n_clones

    new_blocks = []
    new_cds = []
    new_states = []
    new_counties = []

    for c in range(n_clones):
        start = c * n_old
        end = start + n_old
        clone_blocks = geography.block_geoid[start:end]
        clone_cds = geography.cd_geoid[start:end]
        clone_states = geography.state_fips[start:end]
        clone_counties = geography.county_fips[start:end]
        new_blocks.append(np.concatenate([clone_blocks, clone_blocks]))
        new_cds.append(np.concatenate([clone_cds, clone_cds]))
        new_states.append(np.concatenate([clone_states, clone_states]))
        new_counties.append(np.concatenate([clone_counties, clone_counties]))

    return GeographyAssignment(
        block_geoid=np.concatenate(new_blocks),
        cd_geoid=np.concatenate(new_cds),
        state_fips=np.concatenate(new_states),
        county_fips=np.concatenate(new_counties),
        n_records=n_new,
        n_clones=n_clones,
    )
