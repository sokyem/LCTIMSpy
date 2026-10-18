"""Tests for the peptide -> m/z math in ``get_fragmentMz``.

This is the only module with no Bruker SDK dependency, so it is the natural
first thing to pin down before the refactor moves it to ``core/fragments.py``.
"""

import pytest

from tests.conftest import PROTON

PEPTIDE = "PEPTIDE"          # length 7 -> 6 fragments per series
AMIDATED = "PEPTIDE(-0.98)"


def test_series_lengths(fragments):
    """Each series has len(peptide) - 1 fragments."""
    result = fragments.calculate_fragment_mz(PEPTIDE, charge_states=[1])
    assert set(result) == {"b1+", "y1+"}
    assert len(result["b1+"]) == len(PEPTIDE) - 1
    assert len(result["y1+"]) == len(PEPTIDE) - 1


def test_dict_key_is_charge_not_ion_number(fragments):
    """"b2+" means the whole b series at charge 2, not the b2 ion.

    Easy to misread, and the README's `fragments["y1+"]` idiom depends on it.
    """
    result = fragments.calculate_fragment_mz(PEPTIDE, charge_states=[1, 2, 3])
    assert set(result) == {"b1+", "b2+", "b3+", "y1+", "y2+", "y3+"}
    for key in result:
        assert len(result[key]) == len(PEPTIDE) - 1


def test_y_series_is_descending_ion_order(fragments):
    """y[0] is the LONGEST suffix (y6), y[-1] the shortest (y1)."""
    y = fragments.calculate_fragment_mz(PEPTIDE, charge_states=[1])["y1+"]
    assert y == sorted(y, reverse=True)


def test_bn_yn_complementarity(fragments):
    """b[k] + y[k] == neutral + 2 protons, for every k.

    The classic b/y relation, and the strongest single check that the series are
    paired correctly. It only holds because y is in descending ion order.
    """
    result = fragments.calculate_fragment_mz(PEPTIDE, charge_states=[1])
    precursor = fragments.calculate_precursor_mz(PEPTIDE, charge_states=[1])["M1+"]
    expected = precursor + PROTON

    sums = [b + y for b, y in zip(result["b1+"], result["y1+"])]
    for value in sums:
        assert value == pytest.approx(expected, abs=1e-6)


def test_charge_state_scaling(fragments):
    """m/z at charge z is (M1+ + (z-1) * proton) / z."""
    precursor = fragments.calculate_precursor_mz(PEPTIDE, charge_states=[1, 2, 3])
    singly = precursor["M1+"]
    for z in (2, 3):
        expected = (singly + (z - 1) * PROTON) / z
        assert precursor[f"M{z}+"] == pytest.approx(expected, abs=1e-6)


def test_amidation_shifts_y_and_precursor_but_not_b(fragments):
    """C-terminal amidation is a C-term modification, so b ions are unaffected."""
    plain = fragments.calculate_fragment_mz(PEPTIDE, charge_states=[1])
    amid = fragments.calculate_fragment_mz(AMIDATED, charge_states=[1])

    assert amid["b1+"] == pytest.approx(plain["b1+"], abs=1e-9)
    for got, base in zip(amid["y1+"], plain["y1+"]):
        assert got == pytest.approx(base - 0.98, abs=1e-9)

    p_plain = fragments.calculate_precursor_mz(PEPTIDE, charge_states=[1])["M1+"]
    p_amid = fragments.calculate_precursor_mz(AMIDATED, charge_states=[1])["M1+"]
    assert p_amid == pytest.approx(p_plain - 0.98, abs=1e-9)


def test_amidation_shift_scales_with_charge(fragments):
    """The 0.98 neutral loss is divided by z on the m/z scale."""
    plain = fragments.calculate_precursor_mz(PEPTIDE, charge_states=[2])["M2+"]
    amid = fragments.calculate_precursor_mz(AMIDATED, charge_states=[2])["M2+"]
    assert amid == pytest.approx(plain - 0.98 / 2, abs=1e-9)


def test_fragments_to_save_filters(fragments):
    result = fragments.calculate_fragment_mz(
        PEPTIDE, charge_states=[1, 2], fragments_to_save=["y1+", "b2+"]
    )
    assert set(result) == {"y1+", "b2+"}


def test_default_charge_states_not_mutated_between_calls(fragments):
    """`charge_states=[1, 2, 3]` is a mutable default; prove it is not accumulating."""
    first = fragments.calculate_fragment_mz(PEPTIDE)
    second = fragments.calculate_fragment_mz(PEPTIDE)
    assert set(first) == set(second) == {"b1+", "b2+", "b3+", "y1+", "y2+", "y3+"}
