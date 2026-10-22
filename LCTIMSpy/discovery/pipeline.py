"""One discovery run: features -> PSMs -> candidates -> MS/MS validation -> saved result.

A run is saved to its own folder (``DiscoveryStore``)::

    meta.json            config, inputs, counts, timings, capability notes
    candidates.json      every candidate (excluded ones too), JSON-safe
    features.parquet     the MS1 features the mass-first route used
    excluded_psms.csv    PSMs left out, each with its reason
"""

from __future__ import annotations

import json
import math
import re
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.progress import CancelToken, ProgressEvent, ProgressReporter, null_reporter
from .candidates import CLASSES, CandidateConfig, discover
from .evidence import EvidenceConfig, add_data_evidence, summarize_support
from .features import FeatureFinderConfig, find_features
from .msms import MsmsConfig, validate_msms
from .psms import read_psms


@dataclass(frozen=True)
class DiscoveryConfig:
    features: FeatureFinderConfig = field(default_factory=FeatureFinderConfig)
    candidates: CandidateConfig = field(default_factory=CandidateConfig)
    msms: MsmsConfig = field(default_factory=MsmsConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def with_overrides(self, overrides: dict | None) -> DiscoveryConfig:
        """Apply the user-facing settings (flat names) onto the nested configs."""
        o = {k: v for k, v in (overrides or {}).items() if v is not None}
        c, m, f, e = self.candidates, self.msms, self.features, self.evidence
        if "mass_ppm" in o:
            c = replace(c, grouping=replace(c.grouping, mass_ppm=float(o["mass_ppm"])))
            e = replace(e, mass_agreement_ppm=float(o["mass_ppm"]))
        if "isotope_min_cosine" in o:
            e = replace(e, min_isotope_cosine=float(o["isotope_min_cosine"]))
        for key in ("rt_min_delta", "rt_min_resolution", "mobility_min_rel_pct"):
            if key in o:
                c = replace(c, **{key: float(o[key])})
        if "mobility_precision_rel" in o:
            mob = replace(c.trace.mobility, precision_rel=float(o["mobility_precision_rel"]),
                          precision_is_placeholder=False)
            c = replace(c, trace=replace(c.trace, mobility=mob))
        for key in ("similarity_threshold", "min_annotated_fraction"):
            if key in o:
                m = replace(m, **{key: float(o[key])})
        if "min_snr" in o:
            f = replace(f, min_snr=float(o["min_snr"]))
        return replace(self, features=f, candidates=c, msms=m, evidence=e)


def to_jsonable(obj):
    """Recursively convert numpy/pandas values; NaN and inf become None."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if obj is pd.NA or obj is pd.NaT:
        return None
    return obj


def _stem(name: str) -> str:
    base = re.split(r"[\\/]", str(name))[-1]
    for suffix in (".mzml.gz", ".mzml", ".d", ".raw", ".mgf", ".mzxml"):
        if base.lower().endswith(suffix):
            base = base[: -len(suffix)]
    return base.lower()


def select_run_psms(psms: pd.DataFrame, dataset_name: str) -> tuple[pd.DataFrame, str]:
    """The PSMs of this run, matched by run/file name; a note says how."""
    runs = psms["run"].fillna("").astype(str)
    stems = runs.map(_stem)
    distinct = sorted(set(stems) - {""})
    target = _stem(dataset_name)
    if not distinct:
        return psms, "the PSM file names no runs; all PSMs used"
    hit = (stems == target) | stems.str.contains(re.escape(target), regex=True)
    if hit.any():
        return psms[hit], f"{int(hit.sum())} of {len(psms)} PSMs matched run {dataset_name!r}"
    if len(distinct) == 1:
        return psms, (f"the PSM file holds one run ({distinct[0]!r}) that does not match "
                      f"{dataset_name!r}; all PSMs used -- check that this is intended")
    return psms.iloc[0:0], (f"no PSMs for run {dataset_name!r}; the file contains "
                            f"{', '.join(distinct[:5])}{'...' if len(distinct) > 5 else ''}")


def run_discovery(cache, entry, out_dir: Path, psm_path: str | None = None,
                  psm_engine: str | None = None, cfg: DiscoveryConfig = DiscoveryConfig(),
                  reporter: ProgressReporter = null_reporter,
                  cancel: CancelToken | None = None, dataset_name: str | None = None) -> dict:
    cancel = cancel or CancelToken()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    timings = {}
    notes = []

    t = time.perf_counter()
    features, _ = find_features(cache, entry, cfg.features, reporter=reporter, cancel=cancel)
    timings["features_s"] = time.perf_counter() - t

    psms = None
    psm_info = None
    if psm_path:
        reporter(ProgressEvent("psms", 0, 1, "Reading PSMs"))
        all_psms = read_psms(psm_path, psm_engine)
        psms, note = select_run_psms(all_psms, dataset_name or entry.dataset.name)
        notes.append(note)
        psm_info = {"path": str(psm_path), "engine": all_psms.attrs.get("engine"),
                    "total": len(all_psms), "used": len(psms), "note": note}
    cancel.raise_if_cancelled()

    reporter(ProgressEvent("candidates", 0, 1, "Discovering candidates"))
    t = time.perf_counter()
    candidates, excluded = discover(cache, entry, features, psms, cfg.candidates)
    timings["candidates_s"] = time.perf_counter() - t
    cancel.raise_if_cancelled()

    t = time.perf_counter()
    if len(candidates):
        candidates = add_data_evidence(cache, entry, candidates, cfg.evidence, cfg.candidates.trace,
                                       reporter=reporter, cancel=cancel)
    timings["evidence_s"] = time.perf_counter() - t
    cancel.raise_if_cancelled()

    reporter(ProgressEvent("msms", 0, 1, "Validating with MS/MS"))
    t = time.perf_counter()
    if len(candidates):
        candidates = validate_msms(cache, entry, candidates, features, cfg.msms)
        candidates["data_support"] = [summarize_support(r) for r in candidates.to_dict(orient="records")]
    timings["msms_s"] = time.perf_counter() - t
    reporter(ProgressEvent("msms", 1, 1, "Saving"))

    records = to_jsonable(candidates.to_dict(orient="records")) if len(candidates) else []
    (out_dir / "candidates.json").write_text(json.dumps(records), encoding="utf-8")
    features.drop(columns=[c for c in ("isotope_intensities", "trace_ids") if c in features]) \
        .to_parquet(out_dir / "features.parquet", index=False)
    if len(excluded):
        excluded.to_csv(out_dir / "excluded_psms.csv", index=False)

    counts = summarize(records)
    meta = to_jsonable({
        "created": started, "finished": time.time(), "timings": timings,
        "dataset": entry.dataset.to_dict(), "cache_key": entry.key, "mobility": entry.mobility,
        "psms": psm_info, "excluded_psms": len(excluded), "n_features": len(features),
        "n_charged_features": int(features["charge"].notna().sum()) if len(features) else 0,
        "counts": counts, "config": cfg.to_dict(), "notes": notes,
    })
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def summarize(records: list[dict]) -> dict:
    out = {"total": len(records), "candidates": 0, "excluded": 0,
           "by_class": {v: 0 for v in CLASSES.values()}, "by_category": {},
           "by_data_support": {"strong": 0, "moderate": 0, "weak": 0}}
    for r in records:
        out["excluded" if r["status"] == "excluded" else "candidates"] += 1
        if r["status"] != "excluded":
            out["by_class"][r["classification"]] = out["by_class"].get(r["classification"], 0) + 1
            out["by_category"][r["category"]] = out["by_category"].get(r["category"], 0) + 1
            level = (r.get("data_support") or {}).get("level")
            if level:
                out["by_data_support"][level] += 1
    return out


def new_run_id() -> str:
    return uuid.uuid4().hex[:10]
