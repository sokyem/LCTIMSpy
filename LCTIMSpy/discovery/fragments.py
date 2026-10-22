"""b/y fragment ladders for a sequence with its modifications (canonical ``mods``).

``mods`` is the canonical string from ``discovery.psms`` ("3:+14.02;12:-0.98"):
position 0 = N-terminus, 1..n = residue, n+1 = C-terminus. Deltas are given
to 0.01 Da there; that is ample for fragment annotation at 20 ppm on
fragments below ~1500 m/z, and named modifications without a mass are
reported rather than guessed.
"""

from __future__ import annotations

import numpy as np
from pyteomics import mass

PROTON = 1.007276466812
WATER = mass.calculate_mass(formula="H2O")


class UnknownModification(ValueError):
    """A modification or residue without a defined mass: fragments cannot be computed."""


def _deltas(peptide: str, mods: str) -> np.ndarray:
    """Mass added at positions 0..n+1."""
    out = np.zeros(len(peptide) + 2)
    for part in filter(None, (mods or "").split(";")):
        pos, _, delta = part.partition(":")
        try:
            out[int(pos)] += float(delta)
        except ValueError as exc:
            raise UnknownModification(f"modification {part!r} has no mass") from exc
    return out


def fragment_ladder(peptide: str, mods: str = "", charge: int = 1
                    ) -> tuple[np.ndarray, list[str]]:
    """``(mz, labels)`` of b1..b(n-1) and y1..y(n-1) at the given fragment charge."""
    aa = mass.std_aa_mass
    unknown = sorted({r for r in peptide if r not in aa})
    if unknown:
        # Ambiguous or non-standard letters (B, Z, X, J, ...) have no defined mass.
        raise UnknownModification(f"residue(s) {', '.join(unknown)} in {peptide!r} have no mass")
    residues = np.array([aa[r] for r in peptide])
    d = _deltas(peptide, mods)
    residues = residues + d[1:-1]
    nterm, cterm = d[0], d[-1]
    n = len(peptide)
    b = np.cumsum(residues)[:-1] + nterm
    y = np.cumsum(residues[::-1])[:-1] + WATER + cterm
    mz = np.concatenate([(b + charge * PROTON) / charge, (y + charge * PROTON) / charge])
    labels = [f"b{i}" for i in range(1, n)] + [f"y{i}" for i in range(1, n)]
    if charge > 1:
        labels = [f"{label}{'+' * charge}" for label in labels]
    return mz, labels
