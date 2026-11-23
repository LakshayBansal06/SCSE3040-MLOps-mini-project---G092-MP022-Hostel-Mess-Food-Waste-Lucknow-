"""Tests for the PSI drift score used in monitoring."""

import numpy as np
import pandas as pd

from mess_waste.monitor import calculate_psi


def test_psi_is_about_zero_for_the_same_distribution():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(50, 10, 2000))
    b = pd.Series(rng.normal(50, 10, 2000))
    assert calculate_psi(a, b) < 0.05


def test_psi_flags_a_clear_shift():
    rng = np.random.default_rng(1)
    reference = pd.Series(rng.normal(25, 5, 2000))
    shifted = pd.Series(rng.normal(35, 5, 2000))
    assert calculate_psi(reference, shifted) > 0.25


def test_bigger_shift_gives_bigger_psi():
    rng = np.random.default_rng(2)
    reference = pd.Series(rng.normal(0, 1, 2000))
    small = calculate_psi(reference, pd.Series(rng.normal(0.3, 1, 2000)))
    large = calculate_psi(reference, pd.Series(rng.normal(1.5, 1, 2000)))
    assert 0 <= small < large


def test_psi_survives_constant_columns_blanks_and_text():
    constant = pd.Series([5.0] * 100)
    messy = pd.Series([5.0, np.nan, "oops", 6.0] * 25)
    value = calculate_psi(constant, messy)
    assert np.isfinite(value)
    assert value >= 0
