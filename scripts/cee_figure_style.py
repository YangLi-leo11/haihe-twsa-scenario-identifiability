"""Shared publication style for CEE main-text Figures 2-6."""

from __future__ import annotations

from pathlib import Path
import hashlib
import matplotlib as mpl

MM = 1.0 / 25.4
FINAL_WIDTH_MM = 180.0
DPI = 600

FONT_FAMILY = "Arial"
FONT_FALLBACK = "Helvetica"
TEXT = "#222222"
FRAME = "#4E5A60"
GRID = "#D8DEE2"
ZERO_LINE = "#727B80"
BACKGROUND = "#FFFFFF"

PANEL_LETTER = 9.5
AXIS_LABEL = 8.5
TICK_LABEL = 7.5
LEGEND = 7.4
INTERNAL_HEADING = 8.5
ANNOTATION = 7.2
COLORBAR_TICK = 7.1
COLORBAR_TITLE = 7.5

# Readable aliases used by the individual figure builders.
PANEL_SIZE = PANEL_LETTER
AXIS_SIZE = AXIS_LABEL
TICK_SIZE = TICK_LABEL
LEGEND_SIZE = LEGEND
HEADING_SIZE = INTERNAL_HEADING
ANNOTATION_SIZE = ANNOTATION
CBAR_TICK_SIZE = COLORBAR_TICK
CBAR_TITLE_SIZE = COLORBAR_TITLE

SSP = {
    "ssp126": "#3F78B5",
    "ssp245": "#3E9275",
    "ssp370": "#D88A05",
    "ssp585": "#A74D8D",
}
SSP_COLORS = SSP

REFERENCE = "#353C40"
MODEL_OR_CLIMATE = "#27638A"
MODEL = MODEL_OR_CLIMATE
RESIDUAL = "#58AAA5"
REFERENCE_MONTHLY = "#6B7174"
MODEL_MONTHLY = "#6E9EC0"

PREDICTOR = {
    "P": "#557C96",
    "T": "#76936B",
    "ET": "#C59B53",
    "Q": "#897B97",
}

PATHWAY = {
    "persistence": "#D2762D",
    "groundwater_consistency": "#5F877F",
    "moderation": "#9A8446",
    "stationary_residual": "#73777A",
    "risk_conditioned": "#4E73B5",
}
PATHWAY["gw_consistency"] = PATHWAY["groundwater_consistency"]
PATHWAY["stationary"] = PATHWAY["stationary_residual"]

NEG = "#B65D52"
ZERO = "#F5F3EE"
POS = "#5C929A"


def apply_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_FAMILY, FONT_FALLBACK, "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": TICK_LABEL,
        "axes.labelsize": AXIS_LABEL,
        "xtick.labelsize": TICK_LABEL,
        "ytick.labelsize": TICK_LABEL,
        "legend.fontsize": LEGEND,
        "axes.linewidth": 0.8,
        "axes.edgecolor": FRAME,
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.8,
        "ytick.major.size": 2.8,
        "axes.spines.left": True,
        "axes.spines.right": True,
        "axes.spines.top": True,
        "axes.spines.bottom": True,
        "axes.facecolor": BACKGROUND,
        "figure.facecolor": BACKGROUND,
        "savefig.facecolor": BACKGROUND,
        "savefig.transparent": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def style_cartesian(ax, *, grid_axis: str | None = "y") -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(FRAME)
        spine.set_linewidth(0.8)
    ax.tick_params(axis="both", direction="out", length=2.8, width=0.7,
                   labelsize=TICK_LABEL, colors=TEXT)
    ax.grid(False)
    if grid_axis:
        ax.grid(axis=grid_axis, which="major", color=GRID,
                linewidth=0.45, alpha=0.55, zorder=0)
        ax.set_axisbelow(True)


def panel_label(ax, letter: str, *, x: float = 0.012, y: float = 0.975):
    return ax.text(x, y, letter, transform=ax.transAxes, ha="left", va="top",
                   fontsize=PANEL_LETTER, fontweight="bold", color=TEXT,
                   clip_on=False, zorder=100)


def legend_kwargs(**overrides) -> dict[str, object]:
    values = {
        "frameon": False,
        "fontsize": LEGEND,
        "handlelength": 1.8,
        "handletextpad": 0.45,
        "columnspacing": 1.0,
        "labelspacing": 0.35,
        "borderaxespad": 0.0,
    }
    values.update(overrides)
    return values


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_figure(fig, stem: Path) -> None:
    # The canvas is designed at the final 180-mm page width. Avoid a second
    # renderer-dependent crop, which would make physical widths differ across
    # figures with external axis labels.
    fig.savefig(stem.with_suffix(".png"), dpi=DPI, facecolor=BACKGROUND)
    fig.savefig(stem.with_suffix(".pdf"), facecolor=BACKGROUND)
    fig.savefig(stem.with_suffix(".svg"), facecolor=BACKGROUND)
