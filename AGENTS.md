# Repository Agent Instructions

## Calibration Targets

- This repository has two active calibration target surfaces:
  `policyengine_us_data/utils/loss.py` for the legacy loss matrix and
  `policyengine_us_data/db/etl_national_targets.py` for the database-backed
  unified calibration pipeline.
- When adding, removing, or redefining a calibration target, update both
  surfaces in the same change unless the user explicitly scopes the change to
  one surface or the target cannot be represented in one of them.
- If a calibration target intentionally applies to only one surface, document
  why in the code or PR notes and add a regression test for that decision.
- Add or update tests for both surfaces when changing calibration targets.
