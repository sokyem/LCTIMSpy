"""Untargeted discovery of putative peptide isomers in LC-(IMS)-MS/MS data.

Implements and extends the PepSIFinder methodology (Okyem et al., Anal. Chem.
2025): species that share an accurate precursor mass -- and, given enough
evidence, a peptide sequence -- but differ in retention time, ion mobility, or
both. Every result is a *putative* candidate, graded by the strength of its
evidence, until confirmed by structural experiments (synthetic standards,
PIMT, CID-TIMS localization).
"""
