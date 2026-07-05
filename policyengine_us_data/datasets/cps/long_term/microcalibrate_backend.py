"""microcalibrate backend for long-run household weight calibration.

This adopts the shared PolicyEngine calibration library (``microcalibrate``,
which wraps ``l0`` + torch) as a calibration backend for the long-run household
projection, alongside the hand-rolled IPF/entropy/GREG solvers in
``calibration.py``.

Why this exists
---------------
The hand-rolled IPF rake is unbounded: it solves for whatever multiplicative
weight adjustment hits the age controls exactly, with no penalty for moving a
record far from its baseline weight. In the enhanced CPS a small number of
PUF-imputed donor records carry very large per-record income (for example
``miscellaneous_income``) and survive only because the base weights drive them
toward zero. Unbounded raking happily restores them to ordinary weight, which
scrambles income aggregates (baseline income tax then runs to multiples of GDP).

microcalibrate instead learns a per-record *multiplicative* adjustment
initialised at ~1.0 and regularised toward the baseline (noise, optional L0
sparsity), so near-zero donor weights stay near zero. That preserves the income
structure of the base file while still moving the demographics, which is the
behaviour the long-run model needs and the approach peer models (e.g. Yale
Budget Lab) rely on: bounded reweighting from existing weights rather than an
unconstrained re-solve.

The function below mirrors the ``(X, y_target, baseline_weights)`` contract of
the other ``calibrate_*`` solvers so it can be dispatched from
:func:`calibration.calibrate_weights` via ``method="microcalibrate"``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

# Defaults chosen for the ~41k-household x 86-age-target long-run reweight.
DEFAULT_EPOCHS = 256
DEFAULT_LEARNING_RATE = 1e-2
DEFAULT_NOISE_LEVEL = 10.0
DEFAULT_DROPOUT_RATE = 0.0
DEFAULT_L0_LAMBDA = 5e-6


def calibrate_with_microcalibrate(
    X,
    y_target,
    baseline_weights,
    *,
    target_names: Optional[Sequence[str]] = None,
    extra_constraints: Optional[dict] = None,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    noise_level: float = DEFAULT_NOISE_LEVEL,
    dropout_rate: float = DEFAULT_DROPOUT_RATE,
    regularize_with_l0: bool = False,
    l0_lambda: float = DEFAULT_L0_LAMBDA,
    seed: int = 42,
    device: str = "cpu",
):
    """Calibrate household weights to age (and optional extra) targets.

    Args:
        X: Design matrix of shape ``(n_households, n_age_targets)``. Consumed
            directly as microcalibrate's records-by-targets estimate matrix
            (entry ``[h, a]`` is the count of people in age group ``a`` in
            household ``h``).
        y_target: Age target totals, shape ``(n_age_targets,)``.
        baseline_weights: Initial household weights, shape ``(n_households,)``.
        target_names: Optional names for the age target columns.
        extra_constraints: Optional ``{name: (values, target)}`` mapping of extra
            per-household linear constraints (e.g. an aggregate dollar target)
            appended as additional estimate-matrix columns. Demographics-only
            profiles pass nothing here.
        epochs, learning_rate, noise_level, dropout_rate, regularize_with_l0,
        l0_lambda, seed, device: microcalibrate hyperparameters.

    Returns:
        ``(w_new, iterations, info)`` where ``w_new`` is the calibrated weight
        vector, ``iterations`` is the epoch count (for parity with the IPF
        solver's return signature), and ``info`` is a small audit dict.
    """
    from microcalibrate import Calibration

    X = np.asarray(X, dtype=float)
    y_target = np.asarray(y_target, dtype=float)
    baseline_weights = np.asarray(baseline_weights, dtype=float)

    n_targets = X.shape[1]
    if target_names is None:
        names = [f"age_{i}" for i in range(n_targets)]
    else:
        names = [str(name) for name in target_names]

    columns = {names[i]: X[:, i] for i in range(n_targets)}
    targets = [float(v) for v in y_target]

    if extra_constraints:
        for name, (values, target) in extra_constraints.items():
            columns[str(name)] = np.asarray(values, dtype=float)
            targets.append(float(target))

    estimate_matrix = pd.DataFrame(columns)
    target_names_arr = np.array(list(estimate_matrix.columns))
    targets_arr = np.asarray(targets, dtype=float)

    calibration = Calibration(
        weights=baseline_weights,
        targets=targets_arr,
        target_names=target_names_arr,
        estimate_matrix=estimate_matrix,
        epochs=epochs,
        learning_rate=learning_rate,
        noise_level=noise_level,
        dropout_rate=dropout_rate,
        regularize_with_l0=regularize_with_l0,
        l0_lambda=l0_lambda,
        seed=seed,
        device=device,
    )
    calibration.calibrate()
    w_new = np.asarray(calibration.weights, dtype=float)

    summary = calibration.summary()
    max_rel_error = None
    if hasattr(summary, "columns") and "Relative error" in summary.columns:
        max_rel_error = float(summary["Relative error"].abs().max())

    info = {
        "backend": "microcalibrate",
        "epochs": epochs,
        "regularize_with_l0": regularize_with_l0,
        "n_targets": int(estimate_matrix.shape[1]),
        "max_target_rel_error": max_rel_error,
    }
    return w_new, epochs, info
