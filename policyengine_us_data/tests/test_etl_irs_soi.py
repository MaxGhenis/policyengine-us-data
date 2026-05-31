from policyengine_us_data.db.etl_irs_soi import (
    _get_eitc_recipient_constraints,
)


def _constraint_tuples(constraints):
    return {
        (c.constraint_variable, c.operation, c.value)
        for c in constraints
    }


def test_eitc_recipient_constraints_include_positive_eitc():
    constraints = _constraint_tuples(
        _get_eitc_recipient_constraints({"type": "national"})
    )

    assert ("tax_unit_is_filer", "==", "1") in constraints
    assert ("eitc", ">", "0") in constraints


def test_eitc_state_constraints_include_geography():
    constraints = _constraint_tuples(
        _get_eitc_recipient_constraints(
            {"type": "state", "state_fips": 6}
        )
    )

    assert ("tax_unit_is_filer", "==", "1") in constraints
    assert ("eitc", ">", "0") in constraints
    assert ("state_fips", "==", "6") in constraints


def test_eitc_district_constraints_include_geography():
    constraints = _constraint_tuples(
        _get_eitc_recipient_constraints(
            {"type": "district", "congressional_district_geoid": 601}
        )
    )

    assert ("tax_unit_is_filer", "==", "1") in constraints
    assert ("eitc", ">", "0") in constraints
    assert ("congressional_district_geoid", "==", "601") in constraints
