import numpy as np
import pandas as pd
from microimpute.models.qrf import QRF
from policyengine_core.data import Dataset
from typing import Type

from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.datasets.cps.cps import *
from policyengine_us_data.datasets.puf import *
from policyengine_us_data.utils.retirement_limits import (
    get_retirement_limits,
    get_se_pension_limits,
)

# These are sorted by magnitude.
# First 15 contain 90%.
# First 7 contain 75%.
# If you're trying to debug this part of the code and don't want to wait ages
# to see if something breaks, try limiting to those.
IMPUTED_VARIABLES = [
    "employment_income",
    "partnership_s_corp_income",
    "social_security",
    "taxable_pension_income",
    "interest_deduction",
    "tax_exempt_pension_income",
    "long_term_capital_gains",
    "unreimbursed_business_employee_expenses",
    "pre_tax_contributions",
    "taxable_ira_distributions",
    "self_employment_income",
    "w2_wages_from_qualified_business",
    "unadjusted_basis_qualified_property",
    "business_is_sstb",  # bool
    "short_term_capital_gains",
    "qualified_dividend_income",
    "charitable_cash_donations",
    "self_employed_pension_contribution_ald",
    "unrecaptured_section_1250_gain",
    "taxable_unemployment_compensation",
    "taxable_interest_income",
    "domestic_production_ald",
    "self_employed_health_insurance_ald",
    "rental_income",
    "non_qualified_dividend_income",
    "cdcc_relevant_expenses",
    "tax_exempt_interest_income",
    "salt_refund_income",
    "foreign_tax_credit",
    "estate_income",
    "charitable_non_cash_donations",
    "american_opportunity_credit",
    "miscellaneous_income",
    "alimony_expense",
    "farm_income",
    "alimony_income",
    "health_savings_account_ald",
    "non_sch_d_capital_gains",
    "general_business_credit",
    "energy_efficient_home_improvement_credit",
    "amt_foreign_tax_credit",
    "excess_withheld_payroll_tax",
    "savers_credit",
    "student_loan_interest",
    "investment_income_elected_form_4952",
    "early_withdrawal_penalty",
    "prior_year_minimum_tax_credit",
    "farm_rent_income",
    "qualified_tuition_expenses",
    "educator_expense",
    "long_term_capital_gains_on_collectibles",
    "other_credits",
    "casualty_loss",
    "unreported_payroll_tax",
    "recapture_of_investment_credit",
    "deductible_mortgage_interest",
    "qualified_reit_and_ptp_income",
    "qualified_bdc_income",
    "farm_operations_income",
    "estate_income_would_be_qualified",
    "farm_operations_income_would_be_qualified",
    "farm_rent_income_would_be_qualified",
    "partnership_s_corp_income_would_be_qualified",
    "rental_income_would_be_qualified",
    "self_employment_income_would_be_qualified",
]

OVERRIDDEN_IMPUTED_VARIABLES = [
    "partnership_s_corp_income",
    "interest_deduction",
    "unreimbursed_business_employee_expenses",
    "pre_tax_contributions",
    "w2_wages_from_qualified_business",
    "unadjusted_basis_qualified_property",
    "business_is_sstb",
    "charitable_cash_donations",
    "self_employed_pension_contribution_ald",
    "unrecaptured_section_1250_gain",
    "taxable_unemployment_compensation",
    "domestic_production_ald",
    "self_employed_health_insurance_ald",
    "cdcc_relevant_expenses",
    "salt_refund_income",
    "foreign_tax_credit",
    "estate_income",
    "charitable_non_cash_donations",
    "american_opportunity_credit",
    "miscellaneous_income",
    "alimony_expense",
    "health_savings_account_ald",
    "non_sch_d_capital_gains",
    "general_business_credit",
    "energy_efficient_home_improvement_credit",
    "amt_foreign_tax_credit",
    "excess_withheld_payroll_tax",
    "savers_credit",
    "student_loan_interest",
    "investment_income_elected_form_4952",
    "early_withdrawal_penalty",
    "prior_year_minimum_tax_credit",
    "farm_rent_income",
    "qualified_tuition_expenses",
    "educator_expense",
    "long_term_capital_gains_on_collectibles",
    "other_credits",
    "casualty_loss",
    "unreported_payroll_tax",
    "recapture_of_investment_credit",
    "deductible_mortgage_interest",
    "qualified_reit_and_ptp_income",
    "qualified_bdc_income",
    "farm_operations_income",
    "estate_income_would_be_qualified",
    "farm_operations_income_would_be_qualified",
    "farm_rent_income_would_be_qualified",
    "partnership_s_corp_income_would_be_qualified",
    "rental_income_would_be_qualified",
]

# CPS-only variables that should be QRF-imputed for the PUF clone half
# instead of naively duplicated from the CPS donor. These are
# income-correlated variables that exist only in the CPS; demographics,
# IDs, weights, and random seeds are fine to duplicate.
CPS_ONLY_IMPUTED_VARIABLES = [
    # Retirement distributions
    "taxable_401k_distributions",
    "tax_exempt_401k_distributions",
    "taxable_403b_distributions",
    "tax_exempt_403b_distributions",
    "keogh_distributions",
    "taxable_sep_distributions",
    "tax_exempt_sep_distributions",
    # Retirement contributions
    "traditional_401k_contributions",
    "roth_401k_contributions",
    "traditional_ira_contributions",
    "roth_ira_contributions",
    "self_employed_pension_contributions",
    # Social Security sub-components
    "social_security_retirement",
    "social_security_disability",
    "social_security_dependents",
    "social_security_survivors",
    # Transfer income
    "unemployment_compensation",
    "tanf_reported",
    "ssi_reported",
    "child_support_received",
    "veterans_benefits",
    "workers_compensation",
    "disability_benefits",
    "strike_benefits",
    "receives_wic",
    # SPM variables
    "spm_unit_total_income_reported",
    "snap_reported",
    "spm_unit_capped_housing_subsidy_reported",
    "free_school_meals_reported",
    "spm_unit_energy_subsidy_reported",
    "spm_unit_wic_reported",
    "spm_unit_broadband_subsidy_reported",
    "spm_unit_payroll_tax_reported",
    "spm_unit_federal_tax_reported",
    "spm_unit_state_tax_reported",
    "spm_unit_capped_work_childcare_expenses",
    "spm_unit_spm_threshold",
    "spm_unit_net_income_reported",
    "spm_unit_pre_subsidy_childcare_expenses",
    # Medical expenses
    "health_insurance_premiums_without_medicare_part_b",
    "over_the_counter_health_expenses",
    "other_medical_expenses",
    "medicare_part_b_premiums",
    "child_support_expense",
    # Hours/employment
    "weekly_hours_worked",
    "hours_worked_last_week",
    # Previous year income
    "employment_income_last_year",
    "self_employment_income_last_year",
]

# Set versions for O(1) lookup in the concatenation loop.
_OVERRIDDEN_SET = set(OVERRIDDEN_IMPUTED_VARIABLES)
_IMPUTED_SET = set(IMPUTED_VARIABLES)
_CPS_ONLY_SET = set(CPS_ONLY_IMPUTED_VARIABLES)

# Stage-2 income predictors must be in IMPUTED_VARIABLES so
# PUF-imputed values are available.
CPS_STAGE2_INCOME_PREDICTORS = [
    "employment_income",
    "self_employment_income",
    "social_security",
]

# Shared demographic predictors used in both stages.
DEMOGRAPHIC_PREDICTORS = [
    "age",
    "is_male",
    "tax_unit_is_joint",
    "tax_unit_count_dependents",
]

# Stage-1 adds household relationship flags to demographics.
STAGE1_EXTRA_PREDICTORS = [
    "is_tax_unit_head",
    "is_tax_unit_spouse",
    "is_tax_unit_dependent",
]

_QRF_MAX_TRAIN_SAMPLES = 5000


def _to_entity(pred_values, variable_metadata, populations):
    """Map person-level predictions to the variable's entity."""
    entity = variable_metadata.entity.key
    if entity != "person":
        return populations[entity].value_from_first_person(pred_values)
    return pred_values


def apply_retirement_constraints(predictions, X_test, time_period):
    """Enforce IRS contribution limits on retirement variable predictions.

    Args:
        predictions: DataFrame of QRF predictions for retirement
            contribution variables.
        X_test: DataFrame with at least ``age``,
            ``employment_income``, and ``self_employment_income``.
        time_period: Tax year (int) for IRS limit look-up.

    Returns:
        DataFrame with constrained values (same columns).
    """
    limits = get_retirement_limits(time_period)
    se_limits = get_se_pension_limits(time_period)

    age = X_test["age"].values
    catch_up = age >= 50
    emp_income = X_test["employment_income"].values
    se_income = X_test["self_employment_income"].values

    limit_401k = limits["401k"] + catch_up * limits["401k_catch_up"]
    limit_ira = limits["ira"] + catch_up * limits["ira_catch_up"]
    se_pension_cap = np.minimum(
        se_income * se_limits["se_pension_rate"],
        se_limits["se_pension_dollar_limit"],
    )

    # Explicit mapping: variable -> (cap array, zero_mask or None).
    _CONSTRAINT_MAP = {
        "traditional_401k_contributions": (limit_401k, emp_income == 0),
        "roth_401k_contributions": (limit_401k, emp_income == 0),
        "traditional_ira_contributions": (limit_ira, None),
        "roth_ira_contributions": (limit_ira, None),
        "self_employed_pension_contributions": (
            se_pension_cap,
            se_income == 0,
        ),
    }

    result = predictions.clip(lower=0)
    for var in result.columns:
        cap, zero_mask = _CONSTRAINT_MAP.get(var, (None, None))
        if cap is not None:
            result[var] = np.minimum(result[var].values, cap)
        if zero_mask is not None:
            result.loc[zero_mask, var] = 0

    return result


def reconcile_ss_subcomponents(predictions, total_ss):
    """Normalize Social Security sub-components to sum to total.

    Args:
        predictions: DataFrame with columns for each SS
            sub-component (retirement, disability, dependents,
            survivors).
        total_ss: numpy array of total social_security per record.

    Returns:
        DataFrame with reconciled dollar values.
    """
    values = np.maximum(predictions.values, 0)
    row_sums = values.sum(axis=1)
    positive_mask = total_ss > 0

    shares = np.zeros_like(values)
    nonzero_rows = row_sums > 0
    both = positive_mask & nonzero_rows
    shares[both] = values[both] / row_sums[both, np.newaxis]
    # If row_sum == 0 but total_ss > 0, distribute equally.
    equal_rows = positive_mask & ~nonzero_rows
    shares[equal_rows] = 1.0 / values.shape[1]

    out = np.where(
        positive_mask[:, np.newaxis],
        shares * total_ss[:, np.newaxis],
        0.0,
    )
    return pd.DataFrame(out, columns=predictions.columns)


_RETIREMENT_VARS = {
    "traditional_401k_contributions",
    "roth_401k_contributions",
    "traditional_ira_contributions",
    "roth_ira_contributions",
    "self_employed_pension_contributions",
}

_SS_SUBCOMPONENT_VARS = {
    "social_security_retirement",
    "social_security_disability",
    "social_security_dependents",
    "social_security_survivors",
}


def _apply_cps_only_post_processing(
    predictions, X_test, time_period, y_full_imputations
):
    """Apply retirement constraints and SS reconciliation to
    CPS-only QRF predictions.

    Args:
        predictions: DataFrame of all CPS-only QRF predictions.
        X_test: DataFrame with demographic and income features.
        time_period: Tax year (int).
        y_full_imputations: DataFrame of Stage 1 PUF imputations
            (must contain ``social_security`` column).

    Returns:
        Modified predictions DataFrame.
    """
    result = predictions.copy()

    # 1. Retirement constraints
    ret_cols = [c for c in result.columns if c in _RETIREMENT_VARS]
    if ret_cols:
        ret_preds = result[ret_cols]
        constrained = apply_retirement_constraints(
            ret_preds, X_test, time_period
        )
        for col in ret_cols:
            result[col] = constrained[col]

    # 2. Social Security reconciliation
    ss_cols = [
        c for c in result.columns if c in _SS_SUBCOMPONENT_VARS
    ]
    if ss_cols:
        ss_preds = result[ss_cols]
        total_ss = y_full_imputations["social_security"].values
        reconciled = reconcile_ss_subcomponents(ss_preds, total_ss)
        for col in ss_cols:
            result[col] = reconciled[col]

    return result


class ExtendedCPS(Dataset):
    cps: Type[CPS]
    puf: Type[PUF]
    data_format = Dataset.TIME_PERIOD_ARRAYS

    def generate(self):
        from policyengine_us import Microsimulation

        cps_sim = Microsimulation(dataset=self.cps)
        puf_sim = Microsimulation(dataset=self.puf)

        puf_sim.subsample(10_000)

        stage1_predictors = DEMOGRAPHIC_PREDICTORS + STAGE1_EXTRA_PREDICTORS

        # Stage 1a: impute full income set from PUF → CPS.
        puf_train = puf_sim.calculate_dataframe(
            stage1_predictors + IMPUTED_VARIABLES
        )
        cps_test = cps_sim.calculate_dataframe(stage1_predictors)

        qrf = QRF(
            max_train_samples=_QRF_MAX_TRAIN_SAMPLES,
            log_level="INFO",
        )

        y_full_imputations = qrf.fit_predict(
            X_train=puf_train,
            X_test=cps_test,
            predictors=stage1_predictors,
            imputed_variables=IMPUTED_VARIABLES,
            n_jobs=1,
        )

        # Stage 1b: re-impute overridden variables for both halves.
        y_cps_imputations = qrf.fit_predict(
            X_train=puf_train,
            X_test=cps_test,
            predictors=stage1_predictors,
            imputed_variables=OVERRIDDEN_IMPUTED_VARIABLES,
            n_jobs=1,
        )

        del puf_train, cps_test

        # Stage 2: QRF-impute CPS-only variables for PUF clones.
        # Train on CPS using demographics + PUF-imputed income.
        stage2_predictors = (
            DEMOGRAPHIC_PREDICTORS + CPS_STAGE2_INCOME_PREDICTORS
        )
        cps_train = cps_sim.calculate_dataframe(
            stage2_predictors + CPS_ONLY_IMPUTED_VARIABLES
        )

        # PUF clone test data: CPS demographics + imputed income.
        cps_demo = cps_sim.calculate_dataframe(DEMOGRAPHIC_PREDICTORS)
        for var in CPS_STAGE2_INCOME_PREDICTORS:
            if var in y_full_imputations.columns:
                cps_demo[var] = y_full_imputations[var].values
            else:
                cps_demo[var] = cps_sim.calculate(var).values

        y_cps_only_imputations = qrf.fit_predict(
            X_train=cps_train,
            X_test=cps_demo,
            predictors=stage2_predictors,
            imputed_variables=CPS_ONLY_IMPUTED_VARIABLES,
            n_jobs=1,
        )

        y_cps_only_imputations = _apply_cps_only_post_processing(
            y_cps_only_imputations,
            cps_demo,
            self.time_period,
            y_full_imputations,
        )

        del cps_train, cps_demo

        # --- Build extended dataset (CPS + PUF clone) ---
        cps_sim = Microsimulation(dataset=self.cps)
        data = cps_sim.dataset.load_dataset()
        new_data = {}

        for variable in list(data) + IMPUTED_VARIABLES:
            variable_metadata = cps_sim.tax_benefit_system.variables.get(
                variable
            )
            if variable in data:
                values = data[variable][...]
            else:
                values = cps_sim.calculate(variable).values

            if variable in _OVERRIDDEN_SET:
                pred_values = _to_entity(
                    y_cps_imputations[variable].values,
                    variable_metadata,
                    cps_sim.populations,
                )
                values = np.concatenate([pred_values, pred_values])
            elif variable in _IMPUTED_SET:
                pred_values = _to_entity(
                    y_full_imputations[variable].values,
                    variable_metadata,
                    cps_sim.populations,
                )
                values = np.concatenate([values, pred_values])
            elif variable in _CPS_ONLY_SET:
                if variable in y_cps_only_imputations.columns:
                    pred_values = _to_entity(
                        y_cps_only_imputations[variable].values,
                        variable_metadata,
                        cps_sim.populations,
                    )
                    values = np.concatenate([values, pred_values])
                else:
                    values = np.concatenate([values, values])
            elif variable == "person_id":
                values = np.concatenate([values, values + values.max()])
            elif "_id" in variable:
                values = np.concatenate([values, values + values.max()])
            elif "_weight" in variable:
                values = np.concatenate([values, values * 0])
            else:
                values = np.concatenate([values, values])

            new_data[variable] = {
                self.time_period: values,
            }

        self.save_dataset(new_data)


class ExtendedCPS_2024(ExtendedCPS):
    cps = CPS_2024
    puf = PUF_2024
    name = "extended_cps_2024"
    label = "Extended CPS (2024)"
    file_path = STORAGE_FOLDER / "extended_cps_2024.h5"
    time_period = 2024


if __name__ == "__main__":
    ExtendedCPS_2024().generate()
