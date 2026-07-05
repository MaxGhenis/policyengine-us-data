"""Manual production wrapper for long-run CPS projection artifacts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from policyengine_us_data.datasets.cps.long_term.run_household_projection_parallel import (
    parse_years,
)
from policyengine_us_data.utils.data_upload import upload_to_staging_hf
from policyengine_us_data.utils.run_context import resolve_run_id, staging_prefix


SCRIPT_DIR = Path(__file__).resolve().parent
PARALLEL_RUNNER = SCRIPT_DIR / "run_household_projection_parallel.py"
DEFAULT_HF_REPO = "policyengine/policyengine-us-data"
DEFAULT_ARTIFACT_PREFIX = "long_term"
DEFAULT_TAX_ASSUMPTION = "trustees-2025-core-thresholds-v1"
DEFAULT_SUPPORT_AUGMENTATION_START_YEAR = 2075
DEFAULT_SUPPORT_AUGMENTATION_TOP_N_TARGETS = 20
DEFAULT_SUPPORT_AUGMENTATION_DONORS_PER_TARGET = 5
DEFAULT_SUPPORT_AUGMENTATION_MAX_DISTANCE = 3.0
DEFAULT_SUPPORT_AUGMENTATION_CLONE_WEIGHT_SCALE = 0.1
DEFAULT_SUPPORT_AUGMENTATION_SOLVE_MAX_CONSTRAINT_ERROR_PCT = 0.5
DEFAULT_SUPPORT_AUGMENTATION_TOB_DONOR_TOP_N_SOURCES = 500
DEFAULT_SUPPORT_AUGMENTATION_TOB_DONOR_CLONES_PER_SOURCE = 40
DEFAULT_SUPPORT_AUGMENTATION_TOB_DONOR_WEIGHT_SCALE = 0.2
DEFAULT_SUPPORT_AUGMENTATION_TOB_DONOR_WEIGHTING_MODE = "equal_contribution"
DEFAULT_SUPPORT_AUGMENTATION_TOB_DONOR_MIN_WEIGHT_SCALE_MULTIPLIER = 0.1
DEFAULT_SUPPORT_AUGMENTATION_TOB_DONOR_MAX_WEIGHT_SCALE_MULTIPLIER = 10.0
DEFAULT_SUPPORT_AUGMENTATION_BLUEPRINT_BASE_WEIGHT_SCALE = 5.0


def _effective_support_value(
    args: argparse.Namespace,
    name: str,
    default: object,
) -> object | None:
    value = getattr(args, name, None)
    if value is not None:
        return value
    if args.support_augmentation_profile:
        return default
    return None


def _git_sha() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[4],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return completed.stdout.strip()


def _package_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def _add_optional_value(
    command: list[str],
    flag: str,
    value: str | int | float | None,
) -> None:
    if value is None or value == "":
        return
    command.extend([flag, str(value)])


def build_projection_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(PARALLEL_RUNNER),
        "--years",
        args.years,
        "--jobs",
        str(args.jobs),
        "--output-dir",
        str(output_dir),
        "--profile",
        args.profile,
        "--target-source",
        args.target_source,
        "--tax-assumption",
        args.tax_assumption,
    ]
    if args.keep_temp:
        command.append("--keep-temp")

    _add_optional_value(command, "--base-dataset", args.base_dataset)
    _add_optional_value(
        command,
        "--support-augmentation-profile",
        args.support_augmentation_profile,
    )
    _add_optional_value(
        command,
        "--support-augmentation-target-year",
        args.support_augmentation_target_year,
    )
    if args.support_augmentation_align_to_run_year:
        command.append("--support-augmentation-align-to-run-year")
    _add_optional_value(
        command,
        "--support-augmentation-start-year",
        _effective_support_value(
            args,
            "support_augmentation_start_year",
            DEFAULT_SUPPORT_AUGMENTATION_START_YEAR,
        ),
    )
    _add_optional_value(
        command,
        "--support-augmentation-top-n-targets",
        _effective_support_value(
            args,
            "support_augmentation_top_n_targets",
            DEFAULT_SUPPORT_AUGMENTATION_TOP_N_TARGETS,
        ),
    )
    _add_optional_value(
        command,
        "--support-augmentation-donors-per-target",
        _effective_support_value(
            args,
            "support_augmentation_donors_per_target",
            DEFAULT_SUPPORT_AUGMENTATION_DONORS_PER_TARGET,
        ),
    )
    _add_optional_value(
        command,
        "--support-augmentation-max-distance",
        _effective_support_value(
            args,
            "support_augmentation_max_distance",
            DEFAULT_SUPPORT_AUGMENTATION_MAX_DISTANCE,
        ),
    )
    _add_optional_value(
        command,
        "--support-augmentation-clone-weight-scale",
        _effective_support_value(
            args,
            "support_augmentation_clone_weight_scale",
            DEFAULT_SUPPORT_AUGMENTATION_CLONE_WEIGHT_SCALE,
        ),
    )
    _add_optional_value(
        command,
        "--support-augmentation-solve-max-constraint-error-pct",
        _effective_support_value(
            args,
            "support_augmentation_solve_max_constraint_error_pct",
            DEFAULT_SUPPORT_AUGMENTATION_SOLVE_MAX_CONSTRAINT_ERROR_PCT,
        ),
    )
    _add_optional_value(
        command,
        "--support-augmentation-tob-donor-top-n-sources",
        _effective_support_value(
            args,
            "support_augmentation_tob_donor_top_n_sources",
            DEFAULT_SUPPORT_AUGMENTATION_TOB_DONOR_TOP_N_SOURCES,
        ),
    )
    _add_optional_value(
        command,
        "--support-augmentation-tob-donor-clones-per-source",
        _effective_support_value(
            args,
            "support_augmentation_tob_donor_clones_per_source",
            DEFAULT_SUPPORT_AUGMENTATION_TOB_DONOR_CLONES_PER_SOURCE,
        ),
    )
    _add_optional_value(
        command,
        "--support-augmentation-tob-donor-weight-scale",
        _effective_support_value(
            args,
            "support_augmentation_tob_donor_weight_scale",
            DEFAULT_SUPPORT_AUGMENTATION_TOB_DONOR_WEIGHT_SCALE,
        ),
    )
    _add_optional_value(
        command,
        "--support-augmentation-tob-donor-weighting-mode",
        _effective_support_value(
            args,
            "support_augmentation_tob_donor_weighting_mode",
            DEFAULT_SUPPORT_AUGMENTATION_TOB_DONOR_WEIGHTING_MODE,
        ),
    )
    _add_optional_value(
        command,
        "--support-augmentation-tob-donor-min-weight-scale-multiplier",
        _effective_support_value(
            args,
            "support_augmentation_tob_donor_min_weight_scale_multiplier",
            DEFAULT_SUPPORT_AUGMENTATION_TOB_DONOR_MIN_WEIGHT_SCALE_MULTIPLIER,
        ),
    )
    _add_optional_value(
        command,
        "--support-augmentation-tob-donor-max-weight-scale-multiplier",
        _effective_support_value(
            args,
            "support_augmentation_tob_donor_max_weight_scale_multiplier",
            DEFAULT_SUPPORT_AUGMENTATION_TOB_DONOR_MAX_WEIGHT_SCALE_MULTIPLIER,
        ),
    )
    _add_optional_value(
        command,
        "--support-augmentation-blueprint-base-weight-scale",
        _effective_support_value(
            args,
            "support_augmentation_blueprint_base_weight_scale",
            DEFAULT_SUPPORT_AUGMENTATION_BLUEPRINT_BASE_WEIGHT_SCALE,
        ),
    )
    if args.support_augmentation_sanitize_worker_non_target_income:
        command.append("--support-augmentation-sanitize-worker-non-target-income")
    if args.support_augmentation_sanitize_clone_non_target_income:
        command.append("--support-augmentation-sanitize-clone-non-target-income")
    if args.allow_validation_failures:
        command.append("--allow-validation-failures")
    return command


def _artifact_record(path: Path, output_dir: Path, artifact_prefix: str) -> dict:
    rel_path = relative_artifact_path(path, output_dir, artifact_prefix)
    return {
        "local_path": str(path),
        "staging_relative_path": rel_path,
        "size_bytes": path.stat().st_size,
    }


def collect_artifacts(output_dir: Path, artifact_prefix: str) -> list[Path]:
    top_level_patterns = (
        "*.h5",
        "*.h5.metadata.json",
        "calibration_manifest.json",
        "support_augmentation_report*.json",
        "long_run_production_manifest.json",
    )
    artifacts: list[Path] = []
    for pattern in top_level_patterns:
        artifacts.extend(
            path for path in sorted(output_dir.glob(pattern)) if path.is_file()
        )
    log_dir = output_dir / ".parallel_logs"
    if log_dir.exists():
        artifacts.extend(
            path for path in sorted(log_dir.glob("*.log")) if path.is_file()
        )
    return sorted(
        set(artifacts),
        key=lambda path: relative_artifact_path(path, output_dir, artifact_prefix),
    )


def relative_artifact_path(path: Path, output_dir: Path, artifact_prefix: str) -> str:
    artifact_prefix = artifact_prefix.strip("/")
    if path.parent == output_dir / ".parallel_logs":
        return f"{artifact_prefix}/logs/{path.name}"
    return f"{artifact_prefix}/{path.name}"


def write_manifest(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    command: list[str],
    years: list[int],
    run_id: str,
    source_sha: str,
    artifacts: list[Path],
) -> Path:
    manifest_path = output_dir / "long_run_production_manifest.json"
    recorded_artifacts = [path for path in artifacts if path != manifest_path]
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "source_sha": source_sha,
        "git_sha": _git_sha(),
        "github": {
            "repository": os.environ.get("GITHUB_REPOSITORY", ""),
            "workflow": os.environ.get("GITHUB_WORKFLOW", ""),
            "run_id": os.environ.get("GITHUB_RUN_ID", ""),
            "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", ""),
            "run_url": os.environ.get("US_DATA_GITHUB_RUN_URL", ""),
        },
        "package_versions": {
            "policyengine-us-data": _package_version("policyengine_us_data"),
            "policyengine-us": _package_version("policyengine-us"),
            "policyengine-core": _package_version("policyengine-core"),
        },
        "projection": {
            "years_spec": args.years,
            "years": years,
            "jobs": args.jobs,
            "profile": args.profile,
            "target_source": args.target_source,
            "tax_assumption": args.tax_assumption,
            "base_dataset": args.base_dataset or None,
            "allow_validation_failures": args.allow_validation_failures,
            "support_augmentation": {
                "profile": args.support_augmentation_profile or None,
                "target_year": args.support_augmentation_target_year,
                "align_to_run_year": args.support_augmentation_align_to_run_year,
                "start_year": _effective_support_value(
                    args,
                    "support_augmentation_start_year",
                    DEFAULT_SUPPORT_AUGMENTATION_START_YEAR,
                ),
                "top_n_targets": _effective_support_value(
                    args,
                    "support_augmentation_top_n_targets",
                    DEFAULT_SUPPORT_AUGMENTATION_TOP_N_TARGETS,
                ),
                "donors_per_target": _effective_support_value(
                    args,
                    "support_augmentation_donors_per_target",
                    DEFAULT_SUPPORT_AUGMENTATION_DONORS_PER_TARGET,
                ),
                "max_distance": _effective_support_value(
                    args,
                    "support_augmentation_max_distance",
                    DEFAULT_SUPPORT_AUGMENTATION_MAX_DISTANCE,
                ),
                "clone_weight_scale": _effective_support_value(
                    args,
                    "support_augmentation_clone_weight_scale",
                    DEFAULT_SUPPORT_AUGMENTATION_CLONE_WEIGHT_SCALE,
                ),
                "support_solve_max_constraint_error_pct": (
                    _effective_support_value(
                        args,
                        "support_augmentation_solve_max_constraint_error_pct",
                        DEFAULT_SUPPORT_AUGMENTATION_SOLVE_MAX_CONSTRAINT_ERROR_PCT,
                    )
                ),
                "tob_donor_top_n_sources": (
                    _effective_support_value(
                        args,
                        "support_augmentation_tob_donor_top_n_sources",
                        DEFAULT_SUPPORT_AUGMENTATION_TOB_DONOR_TOP_N_SOURCES,
                    )
                ),
                "tob_donor_clones_per_source": (
                    _effective_support_value(
                        args,
                        "support_augmentation_tob_donor_clones_per_source",
                        DEFAULT_SUPPORT_AUGMENTATION_TOB_DONOR_CLONES_PER_SOURCE,
                    )
                ),
                "tob_donor_weight_scale": (
                    _effective_support_value(
                        args,
                        "support_augmentation_tob_donor_weight_scale",
                        DEFAULT_SUPPORT_AUGMENTATION_TOB_DONOR_WEIGHT_SCALE,
                    )
                ),
                "tob_donor_weighting_mode": (
                    _effective_support_value(
                        args,
                        "support_augmentation_tob_donor_weighting_mode",
                        DEFAULT_SUPPORT_AUGMENTATION_TOB_DONOR_WEIGHTING_MODE,
                    )
                ),
                "tob_donor_min_weight_scale_multiplier": (
                    _effective_support_value(
                        args,
                        "support_augmentation_tob_donor_min_weight_scale_multiplier",
                        DEFAULT_SUPPORT_AUGMENTATION_TOB_DONOR_MIN_WEIGHT_SCALE_MULTIPLIER,
                    )
                ),
                "tob_donor_max_weight_scale_multiplier": (
                    _effective_support_value(
                        args,
                        "support_augmentation_tob_donor_max_weight_scale_multiplier",
                        DEFAULT_SUPPORT_AUGMENTATION_TOB_DONOR_MAX_WEIGHT_SCALE_MULTIPLIER,
                    )
                ),
                "blueprint_base_weight_scale": (
                    _effective_support_value(
                        args,
                        "support_augmentation_blueprint_base_weight_scale",
                        DEFAULT_SUPPORT_AUGMENTATION_BLUEPRINT_BASE_WEIGHT_SCALE,
                    )
                ),
                "sanitize_worker_non_target_income": (
                    args.support_augmentation_sanitize_worker_non_target_income
                ),
                "sanitize_clone_non_target_income": (
                    args.support_augmentation_sanitize_clone_non_target_income
                ),
            },
        },
        "command": command,
        "hf_staging": {
            "enabled": args.upload_to_hf_staging,
            "repo": args.hf_repo if args.upload_to_hf_staging else None,
            "artifact_prefix": args.artifact_prefix,
            "run_id": run_id if args.upload_to_hf_staging else None,
        },
        "artifacts": [
            _artifact_record(path, output_dir, args.artifact_prefix)
            for path in recorded_artifacts
        ],
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return manifest_path


def upload_artifacts(
    *,
    artifacts: list[Path],
    output_dir: Path,
    args: argparse.Namespace,
    run_id: str,
    source_sha: str,
) -> int:
    if not os.environ.get("HUGGING_FACE_TOKEN"):
        raise ValueError(
            "HUGGING_FACE_TOKEN is required when --upload-to-hf-staging is set."
        )
    if not run_id:
        raise ValueError(
            "--run-id or US_DATA_RUN_ID is required for HF staging upload."
        )

    files_with_paths = [
        (path, relative_artifact_path(path, output_dir, args.artifact_prefix))
        for path in artifacts
    ]
    return upload_to_staging_hf(
        files_with_paths,
        version=source_sha or "unknown-source",
        hf_repo_name=args.hf_repo,
        batch_size=args.hf_batch_size,
        run_id=run_id,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build long-run CPS projection H5 artifacts and optionally upload "
            "them to a run-scoped Hugging Face staging prefix."
        )
    )
    parser.add_argument("--years", required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", default="ss-payroll-tob")
    parser.add_argument("--target-source", default="trustees_2025_current_law")
    parser.add_argument("--tax-assumption", default=DEFAULT_TAX_ASSUMPTION)
    parser.add_argument("--base-dataset", default="")
    parser.add_argument("--allow-validation-failures", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--support-augmentation-profile", default="")
    parser.add_argument("--support-augmentation-target-year", type=int)
    parser.add_argument(
        "--support-augmentation-align-to-run-year",
        action="store_true",
    )
    parser.add_argument("--support-augmentation-start-year", type=int)
    parser.add_argument("--support-augmentation-top-n-targets", type=int)
    parser.add_argument("--support-augmentation-donors-per-target", type=int)
    parser.add_argument("--support-augmentation-max-distance", type=float)
    parser.add_argument("--support-augmentation-clone-weight-scale", type=float)
    parser.add_argument(
        "--support-augmentation-solve-max-constraint-error-pct",
        type=float,
    )
    parser.add_argument("--support-augmentation-tob-donor-top-n-sources", type=int)
    parser.add_argument("--support-augmentation-tob-donor-clones-per-source", type=int)
    parser.add_argument("--support-augmentation-tob-donor-weight-scale", type=float)
    parser.add_argument(
        "--support-augmentation-tob-donor-weighting-mode",
        choices=["fixed", "equal_contribution"],
    )
    parser.add_argument(
        "--support-augmentation-tob-donor-min-weight-scale-multiplier",
        type=float,
    )
    parser.add_argument(
        "--support-augmentation-tob-donor-max-weight-scale-multiplier",
        type=float,
    )
    parser.add_argument(
        "--support-augmentation-blueprint-base-weight-scale",
        type=float,
    )
    parser.add_argument(
        "--support-augmentation-sanitize-worker-non-target-income",
        action="store_true",
    )
    parser.add_argument(
        "--support-augmentation-sanitize-clone-non-target-income",
        action="store_true",
    )
    parser.add_argument("--run-id", default="")
    parser.add_argument("--source-sha", default="")
    parser.add_argument("--upload-to-hf-staging", action="store_true")
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--hf-batch-size", type=int, default=50)
    parser.add_argument("--artifact-prefix", default=DEFAULT_ARTIFACT_PREFIX)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    years = parse_years(args.years)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = resolve_run_id(args.run_id)
    source_sha = args.source_sha or os.environ.get("GITHUB_SHA", "") or _git_sha()

    command = build_projection_command(args, output_dir)
    print("Running long-run projection command:")
    print(" ".join(command))
    subprocess.run(command, check=True)

    artifacts = collect_artifacts(output_dir, args.artifact_prefix)
    manifest_path = write_manifest(
        args=args,
        output_dir=output_dir,
        command=command,
        years=years,
        run_id=run_id,
        source_sha=source_sha,
        artifacts=artifacts,
    )
    artifacts = collect_artifacts(output_dir, args.artifact_prefix)
    print(f"Wrote production manifest: {manifest_path}")
    print(f"Collected {len(artifacts)} long-run artifacts.")

    if args.upload_to_hf_staging:
        uploaded_count = upload_artifacts(
            artifacts=artifacts,
            output_dir=output_dir,
            args=args,
            run_id=run_id,
            source_sha=source_sha,
        )
        prefix = staging_prefix(run_id, version=source_sha or "unknown-source")
        print(
            f"Uploaded {uploaded_count} files to {prefix}/"
            f"{args.artifact_prefix.strip('/')} in {args.hf_repo}."
        )
    else:
        print("HF staging upload skipped. Pass --upload-to-hf-staging to publish.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
