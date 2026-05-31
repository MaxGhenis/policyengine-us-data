import pandas as pd
from sqlmodel import Session, select

from policyengine_us_data.db import etl_national_targets
from policyengine_us_data.db.create_database_tables import (
    Stratum,
    StratumConstraint,
    Target,
    create_database,
)


def test_load_tax_filer_targets_handles_mixed_reform_ids(
    tmp_path, monkeypatch
):
    storage_dir = tmp_path / "storage"
    db_dir = storage_dir / "calibration"
    db_dir.mkdir(parents=True)
    db_path = db_dir / "policy_data.db"

    engine = create_database(f"sqlite:///{db_path}")
    with Session(engine) as session:
        us_stratum = Stratum(notes="United States")
        session.add(us_stratum)
        session.commit()
        session.refresh(us_stratum)

        filer_stratum = Stratum(
            parent_stratum_id=us_stratum.stratum_id,
            notes="United States - Tax Filers",
        )
        filer_stratum.constraints_rel = [
            StratumConstraint(
                constraint_variable="tax_unit_is_filer",
                operation="==",
                value="1",
            )
        ]
        filer_stratum.targets_rel = [
            Target(
                variable="salt_deduction",
                period=2024,
                reform_id=0,
                value=123.0,
                active=True,
                source="PolicyEngine",
            )
        ]
        session.add(filer_stratum)
        session.commit()

    monkeypatch.setattr(etl_national_targets, "STORAGE_FOLDER", storage_dir)

    tax_filer_df = pd.DataFrame(
        [
            {
                "variable": "salt_deduction",
                "value": 21.247e9,
                "source": "Joint Committee on Taxation",
                "notes": "SALT deduction tax expenditure",
                "year": 2024,
                "reform_id": 1,
            },
            {
                "variable": "income_tax_positive",
                "value": 1.0,
                "source": "Congressional Budget Office",
                "notes": "Positive income tax",
                "year": 2024,
            },
        ]
    )

    etl_national_targets.load_national_targets(
        direct_targets_df=pd.DataFrame(),
        tax_filer_df=tax_filer_df,
        conditional_targets=[],
    )

    with Session(engine) as session:
        raw_salt = session.exec(
            select(Target).where(
                Target.variable == "salt_deduction",
                Target.reform_id == 0,
            )
        ).one()
        expenditure_salt = session.exec(
            select(Target).where(
                Target.variable == "salt_deduction",
                Target.reform_id == 1,
            )
        ).one()
        income_tax = session.exec(
            select(Target).where(Target.variable == "income_tax_positive")
        ).one()

    assert raw_salt.active is False
    assert expenditure_salt.active is True
    assert expenditure_salt.value == 21.247e9
    assert income_tax.reform_id == 0


def test_ssi_stratified_targets_include_age_targets():
    targets = etl_national_targets._ssi_stratified_targets(2024)

    assert len(targets) == 4
    assert {target["variable"] for target in targets} == {"person_count"}
    assert any(
        target["variable"] == "person_count" and target["value"] == 7_289_843
        for target in targets
    )
    assert any(
        target["value"] == 1_001_922
        and target["stratum_notes"]
        == "National SSI Federal Payment Recipients - Age under 18"
        and any(
            constraint["variable"] == "age"
            and constraint["operation"] == "<"
            and constraint["value"] == "18"
            for constraint in target["constraints"]
        )
        for target in targets
    )
    assert not any(
        constraint["variable"] == "social_security"
        for target in targets
        for constraint in target["constraints"]
    )


def test_load_national_targets_handles_stratified_targets(
    tmp_path, monkeypatch
):
    storage_dir = tmp_path / "storage"
    db_dir = storage_dir / "calibration"
    db_dir.mkdir(parents=True)
    db_path = db_dir / "policy_data.db"

    engine = create_database(f"sqlite:///{db_path}")
    with Session(engine) as session:
        us_stratum = Stratum(notes="United States")
        session.add(us_stratum)
        session.commit()

    monkeypatch.setattr(etl_national_targets, "STORAGE_FOLDER", storage_dir)

    stratified_targets = [
        {
            "variable": "person_count",
            "value": 123.0,
            "source": "Unit test",
            "notes": "SSI recipient count",
            "year": 2024,
            "stratum_notes": "National SSI Test Recipients",
            "constraints": [
                {"variable": "ssi", "operation": ">", "value": "0"},
                {"variable": "age", "operation": ">=", "value": "65"},
            ],
        }
    ]

    etl_national_targets.load_national_targets(
        direct_targets_df=pd.DataFrame(),
        tax_filer_df=pd.DataFrame(),
        conditional_targets=[],
        stratified_targets=stratified_targets,
    )

    with Session(engine) as session:
        stratum = (
            session.exec(
                select(Stratum).where(
                    Stratum.notes == "National SSI Test Recipients"
                )
            )
            .unique()
            .one()
        )
        constraints = {
            (
                constraint.constraint_variable,
                constraint.operation,
                constraint.value,
            )
            for constraint in stratum.constraints_rel
        }
        target = session.exec(
            select(Target).where(Target.stratum_id == stratum.stratum_id)
        ).one()

    assert ("ssi", ">", "0") in constraints
    assert ("age", ">=", "65") in constraints
    assert target.variable == "person_count"
    assert target.value == 123.0
