"""Discovery reports: a flat CSV and a standalone HTML summary with definitions and caveats."""

from __future__ import annotations

import html
import io
import time

import pandas as pd

CSV_COLUMNS = [
    "candidate_id", "status", "classification", "category", "route", "mod_key", "charge",
    "neutral_mass", "mono_mz", "delta_rt", "rt_separation", "rt_resolution",
    "mobility_rel_diff_pct", "delta_mobility", "msms_status", "msms_cosine",
    "relative_signal_fraction_a", "sequence_status",
    "a_rt_apex", "a_rt_start", "a_rt_end", "a_mobility", "a_signal", "a_snr", "a_psm_count",
    "b_rt_apex", "b_rt_start", "b_rt_end", "b_mobility", "b_signal", "b_snr", "b_psm_count",
    "flags", "exclusion_reasons",
]

CAVEATS = [
    "All candidates are putative. Categories grade the strength of the available evidence; "
    "they do not establish an isomerization mechanism or the modified residue.",
    "Relative signal fractions are relative LC-MS signal, not molar abundances: ionization "
    "efficiency can differ between isomers, and overlap or suppression affects areas.",
    "Similar fragmentation supports a shared sequence but does not prove identical structures; "
    "different fragmentation does not exclude isomerization.",
    "Screening thresholds (RT, mobility, similarity) are method- and instrument-dependent and "
    "should be set from characterized standards and replicate data.",
    "Synthetic test data validate the software only; they are not evidence about real isomers.",
]


def flatten(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        row = {k: r.get(k) for k in CSV_COLUMNS if not k.startswith(("a_", "b_"))}
        for side, key in (("a", "species_a"), ("b", "species_b")):
            s = r.get(key) or {}
            for f in ("rt_apex", "rt_start", "rt_end", "mobility", "signal", "snr", "psm_count"):
                row[f"{side}_{f}"] = s.get(f)
        row["flags"] = " | ".join(r.get("flags") or [])
        row["exclusion_reasons"] = " | ".join(r.get("exclusion_reasons") or [])
        rows.append(row)
    return pd.DataFrame(rows, columns=CSV_COLUMNS)


def to_csv(records: list[dict]) -> str:
    buf = io.StringIO()
    flatten(records).to_csv(buf, index=False)
    return buf.getvalue()


def _fmt(v, digits=4):
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return html.escape(str(v))


def to_html(meta: dict, records: list[dict]) -> str:
    counts = meta.get("counts", {})
    ds = meta.get("dataset", {})
    cfg = meta.get("config", {})
    cand_cfg = cfg.get("candidates", {})
    mob_cfg = cand_cfg.get("trace", {}).get("mobility", {})
    rows = []
    for r in sorted(records, key=lambda r: (r["status"] != "candidate", r["category"])):
        rows.append(
            "<tr>" + "".join(f"<td>{x}</td>" for x in (
                r["candidate_id"], html.escape(r["status"]), html.escape(r["classification"]),
                html.escape(r["category"]), html.escape(str(r.get("mod_key") or "—")),
                r["charge"], _fmt(r["neutral_mass"]), _fmt(r.get("delta_rt"), 2),
                _fmt(r.get("mobility_rel_diff_pct"), 2), html.escape(r.get("msms_status") or "–"),
                _fmt(r.get("msms_cosine"), 2),
                html.escape("; ".join((r.get("exclusion_reasons") or []) + (r.get("flags") or [])[:3])),
            )) + "</tr>")
    by_class = "".join(f"<li>{html.escape(k)}: {v}</li>" for k, v in counts.get("by_class", {}).items())
    by_cat = "".join(f"<li>{html.escape(k)}: {v}</li>" for k, v in counts.get("by_category", {}).items())
    caveats = "".join(f"<li>{html.escape(c)}</li>" for c in CAVEATS)
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in meta.get("notes", []))
    psm = meta.get("psms") or {}
    created = time.strftime("%Y-%m-%d %H:%M", time.localtime(meta.get("created") or 0))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LCTIMSpy discovery report</title>
<style>
body{{font:14px/1.5 "Segoe UI",system-ui,sans-serif;margin:24px;color:#0f172a;background:#fff}}
h1{{font-size:20px}}h2{{font-size:15px;margin-top:24px}}
table{{border-collapse:collapse;width:100%;font-size:12.5px}}
th,td{{border-bottom:1px solid #e2e8f0;padding:4px 8px;text-align:left;vertical-align:top}}
th{{background:#f1f5f9}}.muted{{color:#475569}}
@media (prefers-color-scheme:dark){{body{{background:#0b1220;color:#e2e8f0}}th{{background:#17223a}}
td,th{{border-color:#23314f}}.muted{{color:#a3b1c6}}}}
</style></head><body>
<h1>Putative isomer candidates — {html.escape(ds.get("name", ""))}</h1>
<p class="muted">Generated {created} by LCTIMSpy. Mobility coordinate: {html.escape(str((meta.get("mobility") or {}).get("coordinate")))}.
PSMs: {html.escape(str(psm.get("engine") or "none"))} ({psm.get("used", 0)} used).</p>
<h2>Summary</h2>
<p>{counts.get("candidates", 0)} candidates, {counts.get("excluded", 0)} excluded (kept with reasons), from {meta.get("n_charged_features", 0)} charged features.</p>
<ul>{by_class}</ul><ul>{by_cat}</ul>
<h2>Settings</h2>
<p class="muted">ΔRT ≥ {cand_cfg.get("rt_min_delta")} min with Rs ≥ {cand_cfg.get("rt_min_resolution")};
mobility difference ≥ {cand_cfg.get("mobility_min_rel_pct")}% (reference: mean of the two population centres);
mobility precision {mob_cfg.get("precision_rel")} {"(placeholder — not established for this instrument)" if mob_cfg.get("precision_is_placeholder") else ""};
mass tolerance {cand_cfg.get("grouping", {}).get("mass_ppm")} ppm.</p>
<h2>Notes</h2><ul>{notes or "<li>none</li>"}</ul>
<h2>Candidates</h2>
<table><thead><tr><th>#</th><th>Status</th><th>Class</th><th>Category</th><th>Sequence</th><th>z</th>
<th>Neutral mass</th><th>ΔRT (min)</th><th>Δ mobility (%)</th><th>MS/MS</th><th>cos</th><th>Reasons / flags</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table>
<h2>Interpretation</h2><ul>{caveats}</ul>
</body></html>"""
