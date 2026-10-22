"""PSM import from PEAKS, MSFragger, MaxQuant, mzIdentML and mzTab into one table.

The fixture files follow each engine's documented export layout. They are
synthetic: real exports from each engine should be added as they become
available, since layouts drift between versions.
"""

import pytest

from LCTIMSpy.discovery.psms import (
    PROTON,
    PSM_COLUMNS,
    PsmFormatError,
    detect_format,
    parse_maxquant_modified,
    parse_msfragger_mods,
    parse_peaks_peptide,
    read_psms,
)

PEAKS_CSV = """Peptide,-10lgP,Mass,Length,ppm,m/z,z,RT,Area,Fraction,Id,Scan,Source File,1/k0 Range,Accession
GD(+14.02)YFDVEW,85.2,1044.42,8,1.2,523.2195,2,41.35,1.2E6,1,101,F1:5521,run1.d,0.90-0.94,P01
GDYFDVEW,90.1,1030.40,8,0.8,516.2115,2,40.10,3.4E6,1,102,F1:5390,run1.d,0.88-0.92,P01
SPQLEDEAKELQ(-0.98),70.4,1385.66,12,-2.0,693.8355,2,55.00,8.0E5,1,103,F1:7710,run1.d,1.02-1.06,P02
"""

MSFRAGGER_TSV = (
    "Spectrum\tSpectrum File\tPeptide\tModified Peptide\tCharge\tRetention\tObserved M/Z\t"
    "Calculated Peptide Mass\tHyperscore\tIon Mobility\tAssigned Modifications\tProtein\n"
    "run1.05521.05521.2\tinterim/run1.mzML\tGDYFDVEW\tGD[129]YFDVEW\t2\t2481.0\t523.2195\t"
    "1044.4230\t35.1\t0.921\t2D(14.0157)\tsp|P01|X\n"
    "run1.09000.09000.3\tinterim/run1.mzML\tPEPTIDEK\t\t3\t1800.0\t310.1600\t927.4500\t20.0\t"
    "0.85\t\trev_sp|P99|Y\n"
)

MAXQUANT_TXT = (
    "Sequence\tModified sequence\tCharge\tm/z\tMass\tMass error [ppm]\tRetention time\t1/K0\t"
    "CCS\tScore\tRaw file\tMS/MS scan number\tLeading razor protein\tReverse\n"
    "GDYFDVEW\t_GD(Methyl (D))YFDVEW_\t2\t523.2195\t1044.4230\t0.5\t41.35\t0.921\t355.2\t120\t"
    "run1\t5521\tP01\t\n"
    "SPQLEDEAKELQ\t_(Acetyl (Protein N-term))SPQLEDEAKELQ_\t2\t714.84\t1427.66\t1.1\t55.0\t1.04\t"
    "410.0\t88\trun1\t7710\tP02\t+\n"
)

MZID = """<?xml version="1.0" encoding="UTF-8"?>
<MzIdentML id="test" version="1.1.0" xmlns="http://psidev.info/psi/pi/mzIdentML/1.1">
 <SequenceCollection>
  <DBSequence id="db1" accession="P01" searchDatabase_ref="sdb1"/>
  <Peptide id="pep1"><PeptideSequence>GDYFDVEW</PeptideSequence>
   <Modification location="2" monoisotopicMassDelta="14.01565"/></Peptide>
  <PeptideEvidence id="pe1" peptide_ref="pep1" dBSequence_ref="db1" isDecoy="false"/>
 </SequenceCollection>
 <DataCollection>
  <Inputs><SpectraData id="sd1" location="run1.mzML"/></Inputs>
  <AnalysisData>
   <SpectrumIdentificationList id="sil1">
    <SpectrumIdentificationResult id="r1" spectrumID="scan=5521" spectraData_ref="sd1">
     <SpectrumIdentificationItem id="i1" rank="1" chargeState="2" experimentalMassToCharge="523.2195"
       calculatedMassToCharge="523.21877" peptide_ref="pep1" passThreshold="true">
      <PeptideEvidenceRef peptideEvidence_ref="pe1"/>
      <cvParam cvRef="PSI-MS" accession="MS:1002354" name="PSM-level q-value" value="0.002"/>
     </SpectrumIdentificationItem>
     <cvParam cvRef="PSI-MS" accession="MS:1000016" name="scan start time" value="2481.0"
       unitCvRef="UO" unitAccession="UO:0000010" unitName="second"/>
     <cvParam cvRef="PSI-MS" accession="MS:1002815" name="inverse reduced ion mobility"
       value="0.921" unitName="volt-second per square centimeter"/>
    </SpectrumIdentificationResult>
   </SpectrumIdentificationList>
  </AnalysisData>
 </DataCollection>
</MzIdentML>
"""

MZTAB = (
    "MTD\tmzTab-version\t1.0.0\n"
    "MTD\tmzTab-mode\tSummary\n"
    "MTD\tmzTab-type\tIdentification\n"
    "MTD\tdescription\ttest\n"
    "MTD\tms_run[1]-location\tfile:///run1.mzML\n"
    "MTD\tpsm_search_engine_score[1]\t[MS, MS:1001330, X!Tandem:expect, ]\n"
    "\n"
    "PSH\tsequence\tPSM_ID\taccession\tunique\tdatabase\tdatabase_version\tsearch_engine\t"
    "search_engine_score[1]\tmodifications\tretention_time\tcharge\texp_mass_to_charge\t"
    "calc_mass_to_charge\tspectra_ref\tpre\tpost\tstart\tend\topt_global_ion_mobility\n"
    "PSM\tGDYFDVEW\t1\tP01\t1\tdb\t1\t[MS, MS:1001476, X!Tandem, ]\t0.001\t2-UNIMOD:34\t2481.0\t"
    "2\t523.2195\t523.21877\tms_run[1]:scan=5521\tK\tR\t10\t17\t0.921\n"
)


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_peaks_parsing_keeps_modification_states_apart(tmp_path):
    df = read_psms(write(tmp_path, "db.psms.csv", PEAKS_CSV))
    assert list(df.columns) == PSM_COLUMNS and len(df) == 3
    methyl, plain, amidated = df.iloc[0], df.iloc[1], df.iloc[2]
    assert methyl.peptide == plain.peptide == "GDYFDVEW"
    assert methyl.mod_key != plain.mod_key                  # PIMT methyl-Asp is its own state
    assert methyl.mods == "2:+14.02" and plain.mods == ""
    assert amidated.mods == "12:-0.98"                       # C-terminal amidation kept
    assert methyl.mobility == pytest.approx(0.92)            # midpoint of the PEAKS range
    assert (methyl.mobility_low, methyl.mobility_high) == (0.90, 0.94)
    assert methyl.rt == pytest.approx(41.35)                 # PEAKS: minutes
    assert methyl.observed_neutral_mass == pytest.approx((523.2195 - PROTON) * 2)


def test_msfragger_units_and_decoys(tmp_path):
    df = read_psms(write(tmp_path, "psm.tsv", MSFRAGGER_TSV))
    target, decoy = df.iloc[0], df.iloc[1]
    assert target.rt == pytest.approx(41.35)                 # 2481 s -> min
    assert target.mobility == pytest.approx(0.921)
    assert target.mods == "2:+14.02"
    assert not target.is_decoy and decoy.is_decoy
    assert target.mass_error_ppm == pytest.approx(
        ((523.2195 - PROTON) * 2 - 1044.4230) / 1044.4230 * 1e6)


def test_maxquant_named_modifications_and_reverse(tmp_path):
    df = read_psms(write(tmp_path, "evidence.txt", MAXQUANT_TXT))
    assert df.iloc[0].mods == "2:+14.02"                     # "Methyl (D)" -> +14.02
    assert df.iloc[1].mods == "0:+42.01"                     # protein N-term acetyl
    assert df.iloc[0].ccs == pytest.approx(355.2)            # engine-derived CCS kept as such
    assert df.iloc[1].is_decoy


def test_mzidentml(tmp_path):
    df = read_psms(write(tmp_path, "search.mzid", MZID))
    row = df.iloc[0]
    assert row.peptide == "GDYFDVEW" and row.mods == "2:+14.02"
    assert row.rt == pytest.approx(41.35) and row.mobility == pytest.approx(0.921)
    assert row.q_value == pytest.approx(0.002) and row.protein == "P01"


def test_mztab(tmp_path):
    df = read_psms(write(tmp_path, "search.mzTab", MZTAB))
    row = df.iloc[0]
    assert row.peptide == "GDYFDVEW" and row.mods == "2:+14.02"
    assert row.rt == pytest.approx(41.35) and row.mobility == pytest.approx(0.921)


def test_same_identification_gets_same_key_in_every_engine(tmp_path):
    keys = {
        read_psms(write(tmp_path, "a.csv", PEAKS_CSV)).iloc[0].mod_key,
        read_psms(write(tmp_path, "b.tsv", MSFRAGGER_TSV)).iloc[0].mod_key,
        read_psms(write(tmp_path, "c.txt", MAXQUANT_TXT)).iloc[0].mod_key,
        read_psms(write(tmp_path, "d.mzid", MZID)).iloc[0].mod_key,
        read_psms(write(tmp_path, "e.mzTab", MZTAB)).iloc[0].mod_key,
    }
    assert keys == {"GDYFDVEW|2:+14.02"}


def test_detection_and_clear_errors(tmp_path):
    assert detect_format(write(tmp_path, "x.csv", PEAKS_CSV)) == "peaks"
    assert detect_format(write(tmp_path, "x.tsv", MSFRAGGER_TSV)) == "msfragger"
    assert detect_format(write(tmp_path, "x.txt", MAXQUANT_TXT)) == "maxquant"
    with pytest.raises(PsmFormatError, match="cannot tell"):
        read_psms(write(tmp_path, "junk.csv", "a,b,c\n1,2,3\n"))
    with pytest.raises(PsmFormatError, match="missing"):
        read_psms(write(tmp_path, "bad.csv", "Peptide,RT\nPEPTIDE,1\n"), engine="peaks")


def test_modification_parsers():
    assert parse_peaks_peptide("K.(+42.01)PEPM(+15.99)IDE.R") == ("PEPMIDE", [(0, "+42.01"), (4, "+15.99")])
    assert parse_msfragger_mods("PEPTIDE", "N-term(42.0106), 7E(-18.0106)") == [(0, "+42.01"), (7, "-18.01")]
    assert parse_maxquant_modified("_PEPM(ox)IDE_") == ("PEPMIDE", [(4, "+15.99")])
    seq, mods = parse_maxquant_modified("_PEP(Unknown thing (X))IDE_")
    assert seq == "PEPIDE" and mods == [(3, "unknown thing")]
