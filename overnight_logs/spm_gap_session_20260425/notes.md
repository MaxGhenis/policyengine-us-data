# SPM poverty gap notes, 2026-04-25

## Benchmarks

- Census benchmark: 2024 Supplemental Poverty Measure, CPS ASEC 2025, all people. Census Table B-3 reports 43.690 million people and a 12.94% SPM poverty rate, rounded to 12.9%.
- Census source page states the 2024 SPM accounts for noncash transfers, geographic housing-cost variation in thresholds, federal/state taxes, work expenses, and medical expenses.
- Local raw Census CPS microdata using household supplement weights reproduces the repo's prior raw baseline: 12.86075% poverty, $101,581.85 average resources, $33,933.56 average threshold. Using SPM person weights gives about 12.91%, closer to Census Table B-3's 12.94%.

## Current local PolicyEngine measurements

- These measurements used local files under `policyengine_us_data/storage`, not the Hugging Face artifact pinned by `policyengine.py`.
- Local artifact checks:
  - `enhanced_cps_2024.h5`: sha256 `53c3fea9c3826f198ed64d5d11d765a6c50fd28169d0475dc10be2b7f92ca417`, 77,006 people.
  - `cps_2024.h5`: sha256 `e2d0167257546bd31af156c0c7e3af4000f8e789f12b508bb71b17083ddb0667`, 325.95M weighted people in the modeled run.
  - `extended_cps_2024_half.h5`: sha256 `6e1c2aacdea42ed5a4eff3adfb642fb67a7ff0e507a8e4ac428c8a7a746c751f`.
- The us-data venv used `policyengine-us==1.637.0`, not the `policyengine.py` US manifest's `policyengine-us==1.653.3`.
- `CPS_2024`, current modeled `person_in_poverty`, 2024: 14.05695%.
- `EnhancedCPS_2024`, current modeled `person_in_poverty`, 2024: 21.96354%.
- `EnhancedCPS_2024`, stored/reported SPM resources vs stored threshold, 2024: 18.03002%.
- Therefore current modeled ECPS is about 9.02 pp above the Census 12.94% benchmark, and about 9.10 pp above the local raw-Census household-weight baseline.

## `policyengine.py` / Hugging Face reproducibility check

- `policyengine.py` currently pins `hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5@1.73.0` with certified artifact sha256 `18cdc668d05311c32ae37364abcea89b0221c27154559667e951c7b19f5b5cbd`, `policyengine-us==1.653.3`, and `policyengine-us-data==1.73.0`.
- The prior local `enhanced_cps_2024.h5` does not match HF main or the `@1.73.0` artifact. HF `@1.73.0` has 101,384 people and size 111,427,742 bytes; HF main has 101,384 people and size 111,878,070 bytes.
- In the `policyengine.py` environment tested here, `resolve_managed_dataset_reference("us")` and `resolve_local_managed_dataset_source("us", ...)` both returned the pinned HF URI, so this specific run did not silently use the local us-data checkout.
- However, `managed_microsimulation()` fails on the pinned `@1.73.0` artifact: `weeks_unemployed` has 192,817 values but the simulation has 101,384 people. The downloaded file's sha256 matches the manifest, so this is a certified-artifact compatibility failure, not a corrupted download.
- Additional stale groups in `@1.73.0` have non-entity lengths and fail us-data's dataset contract validation: `count_under_18`, `count_under_6`, `other_type_retirement_account_distributions`, `reduced_price_school_meals_reported`, `regular_ira_distributions`, and `roth_ira_distributions`.
- HF main is structurally valid under the same `policyengine-us==1.653.3` check: no known entity-length mismatches and no unaligned unknown groups. Its current modeled SPM rate under that package is 23.61672% with 336.67M weighted people.
- A temporary diagnostic copy of the pinned `@1.73.0` artifact with only `weeks_unemployed` removed loads and gives a modeled SPM rate of 23.88065%, but this is not a certified managed result.
- PyPI latest on 2026-04-25: `policyengine==4.3.1`, `policyengine-us==1.667.1`, and `policyengine-us-data==1.88.0`. The local `policyengine.py` source has pyproject version 4.3.1, but the existing venv reports installed distribution metadata `policyengine==0.12.1`, so the env itself also needs a clean sync before using package metadata as provenance.

## Follow-up fix, 2026-04-25

- Updated `policyengine.py`'s US bundle from `policyengine-us==1.653.3` / `enhanced_cps_2024.h5@1.73.0` to `policyengine-us==1.667.1` / `enhanced_cps_2024.h5@1.78.2`.
- `1.78.2` is the newest HF tag found on `policyengine/policyengine-us-data`; PyPI has newer `policyengine-us-data` package versions, but corresponding HF revisions `1.87.0`/`1.88.0` were not published as resolvable tags during this check.
- Fixed `policyengine.py`'s release-bundle refresh helper to stream from the model-style HF resolve URL (`https://huggingface.co/{repo}/resolve/...`) rather than the dataset URL (`https://huggingface.co/datasets/{repo}/resolve/...`), which returned 401 for this repo.
- Removed stale ignored `policyengine.egg-info` in the local `policyengine.py` checkout; `importlib.metadata.version("policyengine")` now reports 4.3.1 in the synced uv environment.
- Verified `managed_microsimulation()` now loads the pinned HF artifact successfully. New managed 2024 SPM poverty rate: 24.25959%, weighted population 334.72M.

## Release/versioning diagnosis, 2026-04-26

- The gap between PyPI `policyengine-us-data==1.88.0` and the newest resolvable HF revision is not just build latency. HF main currently has an untagged `release_manifest.json` for `policyengine-us-data==1.87.0`, created from the main push build, but `hf://policyengine/policyengine-us-data/...@1.87.0` and `@1.88.0` are not resolvable revisions.
- HF main's `1.87.0` manifest tracks only four root artifacts: `cps_2024.h5`, `enhanced_cps_2024.h5`, `policy_data.db`, and `small_enhanced_cps_2024.h5`. It has no `national/`, `states/`, `districts/`, or `cities/` artifacts, so `should_finalize_local_area_release` intentionally refuses to create the stable HF version tag.
- The GitHub `Run Pipeline` workflow for `Update package version` commits only calls `run_pipeline.spawn(...)` and exits; a green GitHub run confirms launch, not pipeline completion or promotion. `run_pipeline` itself ends at `completed` and prints a manual `--action promote` command; promotion is the step that can finally tag the HF release after all required local-area prefixes exist.
- A separate stale-model issue was found and fixed locally: `policyengine-us-data` required only `policyengine-us>=1.637.0`, and Modal images run `uv_sync(frozen=True)`, so the frozen `uv.lock` kept installing `policyengine-us==1.637.0`. HF main's `1.87.0` manifest confirms it was built with `policyengine-us==1.637.0`, despite `policyengine-us==1.667.1` being available.
- Local fix applied in this checkout: bumped the `policyengine-us-data` dependency floor to `policyengine-us>=1.667.1` and refreshed `uv.lock`, which now installs `policyengine-us==1.667.1` and `policyengine-core==3.25.2`. Added a unit guard that fails if the declared `policyengine-us` floor falls behind the locked build version again. Verified with focused upload/reproducibility tests.

## Latest data/model SPM run, 2026-04-26

- `policyengine.py` managed US bundle with installed `policyengine==4.3.1`, `policyengine-us==1.667.1`, and tagged HF data `enhanced_cps_2024.h5@1.78.2`: 2024 SPM poverty rate `24.25959%`; weighted poor `81.20M`; weighted population `334.72M`.
- Latest HF main data (`enhanced_cps_2024.h5` at commit `bf82d31d550b017f30ff78437d6b13ab503de18c`) with installed `policyengine-us==1.667.1`: 2024 SPM poverty rate `23.95805%`; weighted poor `80.66M`; weighted population `336.67M`.
- HF main's manifest still reports data version `1.87.0` built with `policyengine-us==1.637.0`, so the HF-main/latest-model result is useful diagnostically but is not a finalized reproducible release bundle.

## Deeper latest-HF-main decomposition, 2026-04-26

- Base HF `cps_2024.h5` remains close to Census on stored/reported SPM: reported-resource poverty `12.8236%`; modeled poverty with `policyengine-us==1.667.1` is `14.7842%`. This is about a `+2.0 pp` model/resource recomputation effect on base CPS.
- HF main `enhanced_cps_2024.h5` has stored/reported-resource poverty of `20.5322%` before recomputing modeled resources. Modeled poverty is `23.9581%`. So the current ECPS excess vs Census is mostly already in the ECPS weights/data distribution, with model recomputation adding about `+3.43 pp`.
- Transition shares for HF main ECPS under `policyengine-us==1.667.1`: both reported- and modeled-poor `16.93%`; modeled-only poor `7.02%`; reported-only poor `3.60%`; neither `72.44%`.
- For the `modeled_only_poor` group, mean modeled net income is `-$15.9k` while stored reported SPM resources are `$78.0k` and threshold is `$34.3k`. Component identity: pre-tax/pre-expense resources are `-$68.6k` lower than reported, modeled taxes are `$22.5k` higher than reported, and modeled expenses are `$2.8k` higher than the residual reported expense concept. This is not just a tax issue; many modeled-only poor records have modeled market income far below reported total income.
- The most pathological subgroup is modeled-only poor with modeled taxes greater than modeled pre-tax/pre-expense resources: `0.85%` of people. Mean modeled net income is `-$288.0k`, reported resources `$97.8k`, and modeled taxes `$332.4k` vs reported taxes `$28.2k`.
- Weight concentration is the dominant ECPS distribution problem. Base CPS person-equivalent effective sample size is about `13,542`; ECPS is about `831`. Top 100 ECPS SPM units carry `25.4%` of person-equivalent weight; top 500 carry `60.2%`; top 1,000 carry `78.9%`.
- ECPS weights overweight low-resource records relative to CPS. Base CPS reported-resource poverty is `12.82%` weighted and `14.84%` unweighted, with poor records having lower mean weights than nonpoor. ECPS reported-resource poverty is `20.53%` weighted and `16.05%` unweighted, with reported-poor records having 1.35x the mean person-equivalent weight of nonpoor records.
- Prior weight-concentration logs tell the same story. The HF-enhanced run on 2026-04-06 moved reported poverty from `13.43%` in CPS to `20.90%` in enhanced CPS, adding `+7.48 pp`; the `50-100% threshold` band gained `+4.68 pp`, `<50% threshold` gained `+2.80 pp`, and `200%+` lost `-8.42 pp`.
- Latest HF `calibration/logs/unified_run_config.json` reports `n_clones=430`, `n_records=5,159,570`, `weight_nonzero=2,666,585`, `mean_error_pct=4.40`, package version `1.72.3`, branch `calibration-pipeline-improvements`. The diagnostics have large misses on national tax/itemization targets: SALT deduction `+1,119%`, SALT-positive tax-unit count `+808%`, medical-expense-deduction count `+329%`, and positive income-tax count `-39%`. These targets are not SPM targets but can force extreme clone weights and distort low-resource representation.

## Calibration tradeoff check, 2026-04-26

- The reweighting is not categorically wrong: it is doing real work. Using the stored `local_net_worth_100` calibration package to compare pre-fit initial weights with calibrated weights on the production target set, initial weights miss district targets badly (mean absolute relative error `858%`, median `157%`) and national targets extremely badly (mean `21,990%`, median `393%`). After 300 calibration epochs, the same production-set errors fall to district mean `2.6%` and median `1.3%`, state mean `5.0%` and median `2.4%`, and national mean `61.8%` and median `28.0%`.
- The targets that would most clearly fail without calibration are IRS/SOI-style income/tax/capital targets and district dollar targets: national net capital gains, tax-exempt interest, dividends, qualified dividends, SALT, rental income, taxable interest, positive income tax, income tax before credits, AGI, and district self-employment income, real-estate taxes, and pension income. These are precisely the families calibration improves the most.
- But the current high-SPM distribution is still not justified by the existing diagnostics. Even after severe reweighting, national target fit remains weak for SALT deduction, SALT/tax-unit counts, rent, health insurance premiums excluding Medicare Part B, QBI deduction, taxable Social Security, medical expenses, child support, Medicaid, and capped work/childcare expenses. So the evidence says calibration is necessary, but the current sparse/high-concentration solution is not a proven Pareto tradeoff.

## National pipeline calibration tradeoff, 2026-04-26

- The preceding `local_net_worth_100` check is for the local-area unified calibration path, which uses the targets database. The national `enhanced_cps_2024.h5` path uses `policyengine_us_data/utils/loss.py::build_loss_matrix()` and `policyengine_us_data/datasets/cps/enhanced_cps.py::reweight()` instead. The relevant local artifact is the root `calibration_log.csv`.
- On the national pipeline log, the logged pre-fit/epoch-0 weights have only `1.33%` of targets within 10%, mean absolute relative error `82.95%`, and median `76.23%`. The final epoch has `87.14%` within 10%, mean `6.18%`, and median `2.29%`. So national reweighting is absolutely doing real target-fitting work.
- The national target families that would fail most without reweighting include self-employed pension contributions, IRS estate income/losses, JCT tax expenditures, alimony, partnership/S-corp income, taxable return counts, AGI by AGI/status, capital gains losses, rent/royalty losses, business losses, tips, child support, Social Security subtypes, state AGI cells, EITC, total pensions, and total Social Security.
- The national path directly targets several SPM-related or SPM-adjacent constructs: total `spm_unit_spm_threshold`, capped work/childcare expenses, capped housing subsidy, and AGI/counts by SPM-threshold decile. These targets improve sharply from epoch 0 to final: total SPM thresholds `73.5%` error to `4.7%`; capped work/childcare `74.0%` to `2.8%`; capped housing subsidy `76.8%` to `2.6%`; threshold-decile counts move from roughly `63%-82%` errors to mostly `4%-18%`.
- Therefore the next SPM question should be framed as a national-pipeline ablation, not a local-area target-db question: rerun or replay national calibration dropping weak/circular SPM constructs and compare target fit, reported-resource SPM, modeled SPM, and weight concentration.

## National admin-target cleanup patch, 2026-04-26

- The survey/SPM target cleanup had not fully reached the dense national pipeline. `build_loss_matrix()` still consumed hardcoded Census/SPM targets and `spm_threshold_agi.csv`, even though the intended calibration strategy is admin/authoritative component targets rather than Census SPM constructs.
- Local patch applied: removed national hardcoded targets for alimony, child support, non-Part-B medical expenses/premiums, OTC expenses, rent, real-estate taxes, `spm_unit_spm_threshold`, capped housing subsidy, and capped work/childcare expenses; removed SPM-threshold AGI/count decile targets from `build_loss_matrix()`; limited age-bucketed health targets to Medicare Part B; removed the unused ACS-style state real-estate-tax helper from the national loss path.
- Also removed the same survey/SPM direct-sum targets from `db/etl_national_targets.py` and `storage/calibration_targets/pull_hardcoded_targets.py`, with regression tests guarding both the national loss constants and national target ETL.
- Verification: `env -u UV_FROZEN uv run pytest tests/unit/calibration/test_loss_targets.py tests/unit/test_etl_national_targets.py` passed (`15 passed`), and `git diff --check` passed.

## National component-target refinement, 2026-04-26

- The cleanup PR was refined after reviewing rent, property tax, and childcare. We should not remove the whole component families just because the prior targets were survey/SPM constructs.
- Property taxes now have two distinct anchors in PR #830: IRS SOI itemized real-estate-tax amount/count targets for itemizing filers, and ACS 2024 state/national aggregate real-estate taxes for all owner-occupied housing units. These are not equivalent: local SOI gives about $104.8B of itemized deductions for 2024 calibration via the 2022 geography file, while ACS B25090 gives about $370.0B for all owner-occupied real-estate taxes.
- Rent now targets ACS 2024 state aggregate contract rent, annualized from monthly B25060, with a state-sum national total of $764.9B. This matches the `rent` variable more closely than ACS gross rent, because `rent` is built from ACS `RNTP`/contract rent and nets housing assistance in the model.
- Childcare now targets BLS Consumer Expenditure Surveys CE LABSTAT series `CXU670320LB0101M` aggregate expenditure for "Babysitting, childcare, daycare, preschool": $63.092B in 2024. This targets `childcare_expenses`, not the circular Census SPM capped work-childcare deduction. The old capped SPM work-childcare target remains removed.
- Added `acs_housing_costs_2024.csv` and a reproducible Census API refresh script. Updated PR #830 and force-pushed commit `b1e745aa`; focused tests passed (`18 passed`) and `git diff --check` passed. GitHub shows the PR branch mergeable with required `check-fork` queued.

## Private transfer balance target refinement, 2026-04-26

- Follow-up check: after removing gross alimony and child-support targets from the national dense loss, there was no remaining national net-flow target enforcing `paid - received = 0`. The local/unified target DB config still lists gross variables, but that is not the national ECPS path in scope here.
- PR #830 now adds two national accounting targets only: `nation/accounting/alimony_paid_minus_received = 0` and `nation/accounting/child_support_paid_minus_received = 0`. These do not target gross survey levels; they only preserve the closed-system identity that private transfers paid should equal private transfers received in the national population.
- A raw zero target would have been ineffective/pathological because sparse ECPS filtered zero targets and dense reweighting used a relative-error formula with a `+1` target shift. The patch gives these two zero targets explicit $1B absolute-error scales, keeps them through the sparse zero-target filter, and logs scaled errors consistently.
- Verification after this refinement: `env -u UV_FROZEN uv run pytest tests/unit/calibration/test_loss_targets.py tests/unit/test_etl_national_targets.py tests/unit/datasets/test_enhanced_cps_seeding.py` passed (`23 passed`), and `git diff --check` passed.

## Decomposition

- Raw Census -> converted `CPS_2024` stored/reported: 12.86075% -> 13.42840% (+0.57 pp). Resources are identical, but `cps_2024.h5` stored thresholds average $34,545 vs raw Census $33,934 person-weighted. Current `calculate_spm_thresholds_with_geoadj` matches Census `SPM_POVTHRESHOLD` within rounding, so this looks like a stale-H5/data-regeneration issue, not a current formula issue.
- `CPS_2024` current modeled -> `EnhancedCPS_2024` current modeled: 14.05695% -> 21.96354% (+7.91 pp). Average modeled resources fall from $102,870 to $90,312 while thresholds fall from $34,545 to $32,838, so lower resources dominate.
- `EnhancedCPS_2024` stored/reported -> current modeled: 18.03002% -> 21.96354% (+3.93 pp net). Transition shares: 14.30% poor in both, 7.67% modeled-only poor, 3.73% reported-only poor.
- Modeled-only poor group has average modeled net income of -$36,655 despite reported SPM resources of $85,767 and threshold of $32,795. The main mechanical driver is modeled taxes: $48,167 federal tax and $12,145 state tax on $29,917 market income. This is the strongest true-bug signal found in this pass.

## Component evidence

- In current modeled all-person means, enhanced vs CPS has $11.36k lower market income, $1.01k higher benefits, $2.77k higher taxes, and $0.56k lower SPM expenses. Employment income alone is $6.51k lower.
- In current enhanced all-person means, modeled taxes exceed stored reported taxes by about $8.5k/person for federal+state+payroll combined, mostly federal income tax ($18.53k modeled vs $10.34k reported).
- A focused tax-outlier diagnostic found modeled taxes exceed modeled market income plus benefits for 2.02% of person-weighted enhanced records. Within that group, the person-weighted mean excess tax is about $218.7k, so a small set of high-weight tax outliers can materially affect the headline.
- Older component logs show benefits among enhanced poor are much lower than CPS poor for SSI and TANF in the high-weight/HF run, but current local modeled-only poverty is dominated by tax outliers rather than missing benefits.
- Weight concentration remains severe: current local enhanced has 85.3% zero-ish households, effective household sample size about 703, top 100 households carrying 27.3% of weight. This is a major calibration/data stability issue, not a Census definition difference.

## Classification

- Likely true data/model bug: stale stored CPS SPM thresholds in generated H5 files; modeled tax outliers in enhanced modeled-only poor; sparse clone-half weighting producing unstable high-weight cells.
- Missing or weak calibration target: low-/middle-resource distribution using defensible non-circular targets; tax liability by income/state; renter/owner and family-composition stability under sparse weighting.
- Legitimate definitional difference: local raw household-weight benchmark is 12.86% while Census Table B-3 is 12.94%; this is small and mostly weight/universe handling.
- Uncertain: housing subsidy, WIC, TANF, SSI, ACA/PTC, Medicare Part B, and childcare/work expenses remain plausible levers, but this pass found tax and weighting/resource distribution bigger than benefit take-up as explanations of the headline current gap.

## Actionable fixes

1. Regenerate and republish `cps_2024.h5`, `extended_cps_2024*.h5`, and downstream ECPS files with the current SPM threshold calculator, then add a validation check that converted CPS thresholds match Census `SPM_POVTHRESHOLD` within rounding.
2. Add a tax sanity diagnostic for ECPS: flag SPM units where modeled federal+state taxes exceed market income plus benefits, and decompose by tax unit, AGI, filing status, and high-weight cells.
3. Reduce sparse-weight concentration or add guardrails/targets so poverty-critical cells are not carried by a few high-weight households. The `projected_2026_no_h6_snapshot` result at 15.34% suggests concentration choices can move the rate by several points.
4. Add non-circular calibration checks for tax liability by AGI band/state and low-income resource distribution using IRS/SOI, CBO/JCT, SNAP/QC/admin, SSA/SSI, and HHS/TANF administrative sources rather than Census SPM poverty status itself.
5. After tax and weight fixes, revisit benefit/takeup levers: SSI/TANF state structure, SNAP under/over-targeted states, ACA/PTC premium integration, Medicare Part B premium handling, housing subsidy triangulation, and childcare/work expense validation.
