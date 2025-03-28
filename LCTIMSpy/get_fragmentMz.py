
from pyteomics import mass
def calculate_fragment_mz(peptide, charge_states=[1, 2, 3], fragments_to_save=None):
    """
    Calculate b and y ion series for a peptide with given charge states.
    Applies C-terminal amidation correction if the peptide ends with "(-0.98)".

    Parameters
    ----------
    peptide : str
        Peptide sequence. If amidated, it ends with "(-0.98)".
    charge_states : list, optional
        Charge states to consider (default: [1, 2, 3]).
    fragments_to_save : list, optional
        If provided, only include ion types in this list.

    Returns
    -------
    dict
        Dictionary with ion types as keys (e.g. "b1+", "y2+") and lists of m/z values as values.
    """
    amidated = peptide.endswith("(-0.98)")
    clean_peptide = peptide.replace("(-0.98)", "") if amidated else peptide
    n = len(clean_peptide)
    fragment_ions = {}
    for z in charge_states:
        b_series = [mass.calculate_mass(sequence=clean_peptide[:i], ion_type='b', charge=z, monoisotopic=True)
                    for i in range(1, n)]
        if amidated:
            y_series = [mass.calculate_mass(sequence=clean_peptide[i:], ion_type='y', charge=z, monoisotopic=True) - (0.98 / z)
                        for i in range(1, n)]
        else:
            y_series = [mass.calculate_mass(sequence=clean_peptide[i:], ion_type='y', charge=z, monoisotopic=True)
                        for i in range(1, n)]
        fragment_ions[f"b{z}+"] = b_series
        fragment_ions[f"y{z}+"] = y_series
    if fragments_to_save is not None:
        fragment_ions = {k: v for k, v in fragment_ions.items() if k in fragments_to_save}
    return fragment_ions


def calculate_precursor_mz(peptide, charge_states=[1, 2, 3]):
    """
    Calculate precursor m/z values for a peptide for given charge states.
    Accounts for C-terminal amidation if the peptide ends with "(-0.98)".

    Parameters
    ----------
    peptide : str
        Peptide sequence. If amidated, it ends with "(-0.98)".
    charge_states : list of int, optional
        Charge states for calculation (default: [1, 2, 3]).

    Returns
    -------
    dict
        Dictionary with keys "M{charge}+" and corresponding precursor m/z values.
    """
    amidated = peptide.endswith("(-0.98)")
    clean_peptide = peptide.replace("(-0.98)", "") if amidated else peptide
    neutral_mass = mass.calculate_mass(sequence=clean_peptide, ion_type='M', monoisotopic=True)
    if amidated:
        neutral_mass -= 0.98
    precursor_mzs = {}
    for z in charge_states:
        precursor_mz = mass.calculate_mass(sequence=clean_peptide, ion_type='M', charge=z, monoisotopic=True)
        if amidated:
            precursor_mz -= (0.98 / z)
        precursor_mzs[f"M{z}+"] = precursor_mz
    return precursor_mzs