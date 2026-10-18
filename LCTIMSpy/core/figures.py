"""Plotly figure builders. They return ``go.Figure`` and never call ``.show()``.

The legacy plot functions called ``show()`` unconditionally and set
``pio.renderers.default = "browser"`` at import, which in a server opens a
browser on the server. Rendering is the caller's job: the CLI writes HTML, the
web UI receives figure JSON.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import plotly.graph_objects as go

from .config import RtWindow
from .results import ChromatogramSeries, MobilogramSeries, StatsResult
from .stats import significance_stars
from .transforms import normalize

WINDOW_COLORS = ("rgba(31,119,180,0.15)", "rgba(255,127,14,0.15)", "rgba(44,160,44,0.15)",
                 "rgba(214,39,40,0.15)")


def _series_name(target_mz: float, label: str | None) -> str:
    return f"{label} · {target_mz:.4f}" if label else f"m/z {target_mz:.4f}"


def chromatogram_figure(series: Sequence[ChromatogramSeries],
                        windows: Sequence[RtWindow] = (), title: str | None = None,
                        normalized: bool = False) -> go.Figure:
    """Overlaid EICs, with RT windows drawn as shaded bands."""
    fig = go.Figure()
    for s in series:
        y = normalize(s.intensity) if normalized else s.intensity
        fig.add_trace(go.Scatter(x=s.retention_time, y=y, mode="lines",
                                 name=_series_name(s.target_mz, s.label)))
    for i, w in enumerate(windows):
        fig.add_vrect(x0=w.start, x1=w.end, fillcolor=WINDOW_COLORS[i % len(WINDOW_COLORS)],
                      line_width=0, annotation_text=f"W{i + 1}", annotation_position="top left")
    fig.update_layout(title=title, xaxis_title="Retention time (min)",
                      yaxis_title="Normalized intensity" if normalized else "Intensity",
                      hovermode="x unified", template="plotly_white")
    return fig


def mobilogram_figure(series: Sequence[MobilogramSeries], normalized: bool = True,
                      title: str | None = None) -> go.Figure:
    """One trace per (target, window); windows are distinguished by dash style."""
    dashes = ("solid", "dash", "dot", "dashdot")
    fig = go.Figure()
    for s in series:
        y = normalize(s.intensity) if normalized else s.intensity
        fig.add_trace(go.Scatter(
            x=s.mobility, y=y, mode="lines",
            name=f"m/z {s.target_mz:.4f} · W{s.window + 1} ({s.rt_start:g}–{s.rt_end:g} min)",
            line={"dash": dashes[s.window % len(dashes)]},
        ))
    fig.update_layout(title=title, xaxis_title="Mobility 1/K0 (V·s/cm²)",
                      yaxis_title="Normalized intensity" if normalized else "Intensity",
                      template="plotly_white")
    return fig


def mobility_difference_figure(data: pd.DataFrame, metric: str = "mobility_difference",
                               stats: StatsResult | None = None,
                               title: str | None = None) -> go.Figure:
    """Box per target with every replicate shown, plus brackets for significant pairs."""
    fig = go.Figure()
    order = list(dict.fromkeys(data["target_mz"]))
    labels = {t: f"{t:.4f}" for t in order}
    for t in order:
        sub = data[data.target_mz == t]
        fig.add_trace(go.Box(
            y=sub[metric], x=[labels[t]] * len(sub), name=labels[t], boxpoints="all",
            jitter=0.3, pointpos=0, text=sub["sample_id"].astype(str),
            hovertemplate="%{text}: %{y:.5f}<extra></extra>",
        ))
    if stats is not None and not stats.pairs.empty and not data.empty:
        top = float(data[metric].max())
        span = top - float(data[metric].min()) or abs(top) or 1.0
        level = 0
        for row in stats.pairs[stats.pairs.reject].itertuples():
            y = top + span * (0.08 + 0.08 * level)
            a, b = labels.get(row.group_a), labels.get(row.group_b)
            if a is None or b is None:
                continue
            fig.add_shape(type="path", path=f"M {a},{y - span * 0.02} L {a},{y} L {b},{y} "
                                            f"L {b},{y - span * 0.02}",
                          xref="x", yref="y", line={"width": 1})
            fig.add_annotation(x=b, y=y, text=significance_stars(row.adjusted_p_value),
                               showarrow=False, xanchor="left", yanchor="middle")
            level += 1
    fig.update_layout(title=title, xaxis_title="Target m/z",
                      yaxis_title=metric.replace("_", " ").capitalize(),
                      xaxis={"type": "category"}, showlegend=False, template="plotly_white")
    return fig
