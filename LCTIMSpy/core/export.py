"""Writers for tables and figures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .errors import LCTIMSError

TABLE_SUFFIXES = {".csv", ".tsv"}
FIGURE_SUFFIXES = {".html", ".json", ".png", ".svg", ".pdf"}


def write_table(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    if path.suffix not in TABLE_SUFFIXES:
        raise ValueError(f"unsupported table format {path.suffix!r}; use .csv or .tsv")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, sep="\t" if path.suffix == ".tsv" else ",")
    return path


def write_figure(fig, path: str | Path) -> Path:
    """Write a plotly figure. HTML embeds plotly.js so it opens offline."""
    path = Path(path)
    suffix = path.suffix
    if suffix not in FIGURE_SUFFIXES:
        raise ValueError(f"unsupported figure format {suffix!r}; use one of "
                         f"{', '.join(sorted(FIGURE_SUFFIXES))}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".html":
        fig.write_html(path, include_plotlyjs=True, full_html=True)
    elif suffix == ".json":
        fig.write_json(path)
    else:
        try:
            fig.write_image(path)
        except (ImportError, ValueError, RuntimeError) as exc:
            raise LCTIMSError(
                f"Writing {suffix} images needs the kaleido package ({exc}). "
                "Install it with: pip install kaleido"
            ) from exc
    return path
