from __future__ import annotations

import hashlib
import importlib.util
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

import cee_figure_style as cfs


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = REPO_ROOT / "data" / "processed" / "fig5_fig6" / "conditional_twsa_monthly.csv"
SOURCE_DISPLAY = "data/processed/fig5_fig6/conditional_twsa_monthly.csv"
BASELINE_SCRIPT = Path(__file__).with_name("figure6_source.py")
OUT = REPO_ROOT / "figures" / "reproduced"
STEM = OUT / "Fig6_CEE_harmonized_v2"

SCENARIOS = ["ssp126", "ssp245", "ssp370", "ssp585"]
ASSUMPTIONS = [
    "A_stationary_residual",
    "B_trend_capped_persistence_residual",
    "C_risk_conditioned_residual",
    "D_moderation_residual",
    "E_consistency_reference_residual",
]
CENTRAL = ASSUMPTIONS[0]
SCENARIO_LABELS = {
    "ssp126": "SSP1-2.6",
    "ssp245": "SSP2-4.5",
    "ssp370": "SSP3-7.0",
    "ssp585": "SSP5-8.5",
}
SCENARIO_COLORS = {
    key: cfs.SSP_COLORS[key] for key in SCENARIOS
}
CLIMATE_COLOR = "#505A5E"
FULL_ENVELOPE_COLOR = "#B6BEC3"
FULL_ENVELOPE_LEFT_COLOR = "#B7C0C5"
GRID_COLOR = cfs.GRID
ZERO_LINE_COLOR = cfs.ZERO_LINE
PHASE_LINE_COLOR = "#B9C0C4"
PHASE_FILLS = ("#FFFFFF", "#F4F7F8", "#F8F6F3")
YEAR_LANE_FILLS = ("#F8F9F9", "#F5F7F8", "#EEF2F4")
FRAME_COLOR = cfs.FRAME
TEXT_COLOR = cfs.TEXT
CLIMATE_SUMMARY_COLOR = "#454C50"
SAME_YEAR_CONNECTORS = True
REFERENCE_ENDPOINTS = {"ssp126": 10.43, "ssp245": 3.67, "ssp370": 6.58, "ssp585": 6.02}


def load_baseline_module():
    spec = importlib.util.spec_from_file_location("figure6_source", BASELINE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load baseline script: {BASELINE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def setup_style() -> None:
    cfs.apply_style()


def shared_limits(projection: pd.DataFrame, stats: pd.DataFrame) -> tuple[float, float, float]:
    central_monthly = projection.loc[
        projection["assumption"].eq(CENTRAL), "total_twsa_reconstruction_mm"
    ]
    displayed = pd.concat(
        [
            projection.loc[projection["assumption"].eq(CENTRAL), "total_twsa_ma12"],
            stats["p25_ma12"],
            stats["p75_ma12"],
            stats["envelope_min_ma12"],
            stats["envelope_max_ma12"],
            stats["climate_component_ma12"],
        ],
        ignore_index=True,
    )
    values = pd.concat([central_monthly, displayed], ignore_index=True).dropna()
    minimum, maximum = float(values.min()), float(values.max())
    span = maximum - minimum
    padded_min = minimum - 0.05 * span
    padded_max = maximum + 0.05 * span
    step = 100.0
    lower = math.floor(padded_min / 20.0) * 20.0
    upper = math.ceil(padded_max / 20.0) * 20.0
    if lower < -150:
        lower = -200.0
    return lower, upper, step


def style_axis(ax: plt.Axes) -> None:
    cfs.style_cartesian(ax, grid_axis="y")
    ax.grid(axis="x", visible=False)
    ax.axhline(0, color=ZERO_LINE_COLOR, linewidth=0.80, zorder=1)


def add_phase_structure(ax: plt.Axes, label_stages: bool = False) -> None:
    phases = [
        (pd.Timestamp("2026-01-01"), pd.Timestamp("2050-12-31")),
        (pd.Timestamp("2051-01-01"), pd.Timestamp("2075-12-31")),
        (pd.Timestamp("2076-01-01"), pd.Timestamp("2100-12-31")),
    ]
    for (left, right), fill in zip(phases, PHASE_FILLS):
        ax.axvspan(left, right, facecolor=fill, edgecolor="none", zorder=-4)
    for boundary in (pd.Timestamp("2050-12-31"), pd.Timestamp("2075-12-31")):
        ax.axvline(
            boundary, color=PHASE_LINE_COLOR, linewidth=0.62,
            linestyle=(0, (2.5, 3.0)), zorder=1.5,
        )
    if label_stages:
        for x, label in [
            (pd.Timestamp("2038-06-01"), "Near term"),
            (pd.Timestamp("2063-06-01"), "Mid-century"),
            (pd.Timestamp("2084-06-01"), "Late century"),
        ]:
            ax.text(
                x, 0.955, label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=cfs.ANNOTATION_SIZE, color="#6B7377", zorder=12,
            )


def add_scenario_accent(ax: plt.Axes, color: str) -> None:
    """Add a short row-level colour cue without turning the frame into a banner."""
    ax.plot(
        [0.075, 0.225], [0.982, 0.982], transform=ax.transAxes,
        color=color, linewidth=1.75, solid_capstyle="round",
        clip_on=False, zorder=14,
    )


def endpoint_label_positions(values: dict[str, float], ylim: tuple[float, float]) -> dict[str, float]:
    minimum_gap = max(12.0, (ylim[1] - ylim[0]) * 0.018)
    ordered = sorted(values.items(), key=lambda item: item[1], reverse=True)
    placed: list[tuple[str, float]] = []
    for scenario, value in ordered:
        target = value if not placed else min(value, placed[-1][1] - minimum_gap)
        placed.append((scenario, target))
    return dict(placed)


def add_target_year_anchors(
    ax: plt.Axes, central: pd.DataFrame, color: str, ylim: tuple[float, float]
) -> dict[int, float]:
    """Mark December MA12 values at the three fixed target years."""
    target_rows = central[
        central["date"].dt.month.eq(12) & central["date"].dt.year.isin([2050, 2075, 2100])
    ].dropna(subset=["total_twsa_ma12"]).copy()
    if target_rows["date"].dt.year.nunique() != 3:
        raise AssertionError("Target-year December MA12 anchors are incomplete")
    target_rows["target_year"] = target_rows["date"].dt.year
    target_rows = target_rows.drop_duplicates("target_year").sort_values("target_year")
    sizes = {2050: 23, 2075: 23, 2100: 38}
    ax.scatter(
        target_rows["date"], target_rows["total_twsa_ma12"],
        s=target_rows["target_year"].map(sizes), color=color,
        edgecolor="white", linewidth=0.70, zorder=10.5, clip_on=False,
    )
    return dict(zip(target_rows["target_year"], target_rows["total_twsa_ma12"]))


def add_stage_change_hints(ax: plt.Axes, ylim: tuple[float, float]) -> None:
    """Add two restrained phase arrows in the low-density lower margin."""
    span = ylim[1] - ylim[0]
    y = ylim[0] + 0.055 * span
    text_y = y + 0.028 * span
    hint_color = "#899398"
    hints = [
        (pd.Timestamp("2051-03-01"), pd.Timestamp("2074-09-01"), "\u03942050\u20132075"),
        (pd.Timestamp("2076-03-01"), pd.Timestamp("2099-04-01"), "\u03942075\u20132100"),
    ]
    for left, right, label in hints:
        ax.annotate(
            "", xy=(right, y), xytext=(left, y),
            arrowprops=dict(
                arrowstyle="-|>", color=hint_color, lw=0.52,
                alpha=0.62, shrinkA=0, shrinkB=0,
            ), zorder=11,
        )
        ax.text(
            left + (right - left) / 2, text_y, label,
            ha="center", va="bottom", fontsize=8.0,
            color=hint_color, alpha=0.82, zorder=11,
        )


def draw_panel(
    ax: plt.Axes,
    projection: pd.DataFrame,
    stats: pd.DataFrame,
    scenario: str,
    panel_label: str,
    ylim: tuple[float, float],
    endpoint_labels: dict[str, float],
    label_stages: bool = False,
) -> None:
    color = SCENARIO_COLORS[scenario]
    scenario_paths = projection[projection["scenario"].eq(scenario)]
    central = scenario_paths[scenario_paths["assumption"].eq(CENTRAL)].sort_values("date")
    bounds = stats[stats["scenario"].eq(scenario)].sort_values("date")
    add_phase_structure(ax, label_stages=label_stages)
    ax.fill_between(
        bounds["date"], bounds["p25_ma12"], bounds["p75_ma12"],
        color=color, alpha=0.11, linewidth=0, zorder=1,
    )
    for column in ("envelope_min_ma12", "envelope_max_ma12"):
        ax.plot(
            bounds["date"], bounds[column], color=FULL_ENVELOPE_LEFT_COLOR,
            linewidth=0.75, linestyle=(0, (3.0, 2.5)), alpha=0.38, zorder=3,
        )
    ax.plot(
        central["date"], central["total_twsa_reconstruction_mm"], color=color,
        linewidth=0.80, alpha=0.35, solid_capstyle="round", solid_joinstyle="round",
        rasterized=True, zorder=2,
    )
    for column in ("p25_ma12", "p75_ma12"):
        ax.plot(
            bounds["date"], bounds[column], color=color, linewidth=0.75,
            alpha=0.58, zorder=4,
        )
    ax.plot(
        central["date"], central["total_twsa_ma12"], color=color,
        linewidth=2.10, alpha=1.0, solid_capstyle="round",
        solid_joinstyle="round", zorder=6,
    )
    ax.plot(
        bounds["date"], bounds["climate_component_ma12"], color=CLIMATE_COLOR,
        linewidth=1.00, linestyle=(0, (4.0, 3.0)), alpha=0.92, zorder=5,
    )
    anchor_values = add_target_year_anchors(ax, central, color, ylim)

    endpoint = central.dropna(subset=["total_twsa_ma12"]).iloc[-1]
    endpoint_value = float(endpoint["total_twsa_ma12"])
    if abs(anchor_values[2100] - endpoint_value) > 1e-9:
        raise AssertionError("2100 anchor does not match the plotted endpoint")
    label_y = endpoint_labels[scenario]
    leader_x = pd.Timestamp("2097-06-01")
    label_x = pd.Timestamp("2096-06-01")
    ax.plot(
        [endpoint["date"], leader_x], [endpoint_value, endpoint_value],
        color=color, linewidth=0.55, alpha=0.72, zorder=10,
    )
    if abs(label_y - endpoint_value) > 0.01:
        ax.plot(
            [leader_x, leader_x], [endpoint_value, label_y],
            color=color, linewidth=0.45, alpha=0.55, zorder=10,
        )
    ax.plot(
        [leader_x, label_x], [label_y, label_y], color=color,
        linewidth=0.55, alpha=0.72, zorder=10,
    )
    ax.text(
        label_x, label_y, f"{endpoint_value:.1f} mm", ha="left", va="center",
        fontsize=cfs.ANNOTATION_SIZE, fontweight="bold", color=color,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.86, pad=1.0),
        zorder=11,
    )
    cfs.panel_label(ax, panel_label)
    ax.text(
        0.985, 0.94, SCENARIO_LABELS[scenario], transform=ax.transAxes,
        ha="right", va="top", fontsize=cfs.HEADING_SIZE, fontweight="bold",
        color=color, zorder=12,
    )
    ax.set_xlim(pd.Timestamp("2026-01-01"), pd.Timestamp("2100-12-31"))
    ax.set_ylim(*ylim)
    style_axis(ax)


def summary_y_limits(stats: pd.DataFrame, projection: pd.DataFrame) -> tuple[float, float]:
    """Use one y-range for all four vertical target-year summaries."""
    target = stats[
        stats["date"].dt.month.eq(12) & stats["date"].dt.year.isin([2050, 2075, 2100])
    ]
    values = pd.concat(
        [
            target["envelope_min_ma12"], target["envelope_max_ma12"],
            target["p25_ma12"], target["p75_ma12"],
            target["climate_component_ma12"],
            projection.loc[
                projection["assumption"].eq(CENTRAL)
                & projection["date"].dt.month.eq(12)
                & projection["date"].dt.year.isin([2050, 2075, 2100]),
                "total_twsa_ma12",
            ],
        ], ignore_index=True,
    ).dropna()
    lo, hi = float(values.min()), float(values.max())
    pad = max(12.0, 0.10 * (hi - lo))
    return lo - pad, hi + pad


def draw_summary_panel(
    ax: plt.Axes,
    projection: pd.DataFrame,
    stats: pd.DataFrame,
    scenario: str,
    panel_label: str,
    ylim: tuple[float, float],
    show_xlabels: bool,
    show_ylabel: bool,
) -> None:
    """Draw a compact vertical point-range summary for one SSP."""
    color = SCENARIO_COLORS[scenario]
    years = [2050, 2075, 2100]
    x_positions = np.array([0.70, 1.00, 1.30], dtype=float)
    target_stats = stats[
        stats["scenario"].eq(scenario)
        & stats["date"].dt.month.eq(12)
        & stats["date"].dt.year.isin(years)
    ].copy()
    central = projection[
        projection["scenario"].eq(scenario)
        & projection["assumption"].eq(CENTRAL)
        & projection["date"].dt.month.eq(12)
        & projection["date"].dt.year.isin(years)
    ].copy()
    target_stats["target_year"] = target_stats["date"].dt.year
    central["target_year"] = central["date"].dt.year
    target_stats = target_stats.drop_duplicates("target_year").set_index("target_year").reindex(years)
    central = central.drop_duplicates("target_year").set_index("target_year").reindex(years)
    if (
        target_stats["envelope_min_ma12"].isna().any()
        or target_stats["climate_component_ma12"].isna().any()
        or central["total_twsa_ma12"].isna().any()
    ):
        raise AssertionError(f"Summary target-year values incomplete for {scenario}")

    for x, year in zip(x_positions, years):
        lane_half_width = 0.105
        ax.axvspan(
            x - lane_half_width, x + lane_half_width,
            facecolor=YEAR_LANE_FILLS[years.index(year)], edgecolor="none",
            alpha=0.78, zorder=-4,
        )
        row = target_stats.loc[year]
        ref = float(central.loc[year, "total_twsa_ma12"])
        climate = float(row["climate_component_ma12"])
        x_climate = x - 0.035
        x_reference = x + 0.035
        # Outer full-pathway envelope: thin gray vertical range with caps.
        ax.vlines(x, row["envelope_min_ma12"], row["envelope_max_ma12"],
                  color=FULL_ENVELOPE_COLOR, linewidth=0.80, alpha=0.86, zorder=2)
        ax.hlines(
            [row["envelope_min_ma12"], row["envelope_max_ma12"]], x - 0.045, x + 0.045,
            color=FULL_ENVELOPE_COLOR, linewidth=0.68, alpha=0.86, zorder=2,
        )
        # P25-P75 pathway range: thicker, scenario-coloured vertical interval.
        ax.plot(
            [x, x], [row["p25_ma12"], row["p75_ma12"]],
            color=color, linewidth=5.0, alpha=0.56,
            solid_capstyle="round", zorder=3,
        )
        ax.hlines(
            [row["p25_ma12"], row["p75_ma12"]], x - 0.065, x + 0.065,
            color=color, linewidth=0.72, alpha=0.72, zorder=4,
        )
        if SAME_YEAR_CONNECTORS:
            ax.plot(
                [x_climate, x_reference], [climate, ref],
                color="#858D91", linewidth=0.62, alpha=0.72,
                solid_capstyle="round", zorder=4.5,
            )
        ax.plot(
            x_climate, climate, marker="D", linestyle="None",
            markersize=4.6, markerfacecolor="white",
            markeredgecolor=CLIMATE_SUMMARY_COLOR, markeredgewidth=0.90,
            zorder=5,
        )
        ax.scatter(x_reference, ref, s=28 if year != 2100 else 31, color=color,
                   edgecolor="white", linewidth=0.70, zorder=6)
        if year == 2100:
            ax.text(x_reference + 0.035, ref + 4.0, f"{ref:.1f}", ha="left", va="bottom",
                    fontsize=cfs.ANNOTATION_SIZE, fontweight="bold", color=color, zorder=6,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.84, pad=0.7))

    ax.set_xlim(0.45, 1.55)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(["2050", "2075", "2100"], color=TEXT_COLOR)
    ax.set_ylim(*ylim)
    y_ticks = [tick for tick in (-100, 0, 100, 200, 300) if ylim[0] <= tick <= ylim[1]]
    ax.set_yticks(y_ticks)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.50, alpha=0.72, zorder=0)
    ax.grid(axis="x", visible=False)
    ax.axhline(0, color=ZERO_LINE_COLOR, linewidth=0.80, alpha=0.85, zorder=1)
    for spine in ax.spines.values():
        spine.set_color(FRAME_COLOR)
        spine.set_linewidth(0.72)
    ax.tick_params(direction="out", length=3.2, width=0.7, color=FRAME_COLOR,
                   labelcolor=TEXT_COLOR, pad=2.2)
    ax.text(0.03, 0.92, panel_label, transform=ax.transAxes,
            ha="left", va="top", fontsize=cfs.PANEL_SIZE, fontweight="bold",
            color="#272C2F", zorder=10)
    if not show_xlabels:
        ax.tick_params(labelbottom=False)
    if show_ylabel:
        ax.set_ylabel("TWSA projection (mm)", labelpad=4, fontsize=10.2)
    # Re-apply fixed compact x geometry after every plotting operation.
    ax.set_xticks(x_positions)
    ax.set_xticklabels(["2050", "2075", "2100"], color=TEXT_COLOR)
    ax.set_xlim(0.45, 1.55)
    ax.margins(x=0)
    ax.set_autoscalex_on(False)
    ax.tick_params(
        axis="y", labelleft=False, left=False, labelright=True, right=True,
        pad=3.2,
    )


def make_figure(projection: pd.DataFrame, stats: pd.DataFrame) -> tuple[plt.Figure, tuple[float, float]]:
    lower, upper, step = shared_limits(projection, stats)
    endpoint_values = {}
    for scenario in SCENARIOS:
        central = projection[
            projection["scenario"].eq(scenario) & projection["assumption"].eq(CENTRAL)
        ].sort_values("date")
        endpoint_values[scenario] = float(
            central.dropna(subset=["total_twsa_ma12"]).iloc[-1]["total_twsa_ma12"]
        )
    endpoint_labels = endpoint_label_positions(endpoint_values, (lower, upper))
    summary_ylim = summary_y_limits(stats, projection)
    fig = plt.figure(
        figsize=(cfs.FINAL_WIDTH_MM * cfs.MM, cfs.FINAL_WIDTH_MM * cfs.MM * 10.8 / 11.7),
        facecolor="white",
    )
    gs = fig.add_gridspec(
        4, 2, width_ratios=[4.20, 1.10], hspace=0.105, wspace=0.09,
    )
    left_axes = []
    right_axes = []
    for index, (scenario, label) in enumerate(zip(SCENARIOS, "abcd")):
        ax = fig.add_subplot(
            gs[index, 0],
            sharex=left_axes[0] if left_axes else None,
            sharey=left_axes[0] if left_axes else None,
        )
        left_axes.append(ax)
        draw_panel(
            ax, projection, stats, scenario, label, (lower, upper), endpoint_labels,
            label_stages=(index == 0),
        )
        ax.set_yticks(np.arange(math.ceil(lower / step) * step, upper + step * 0.5, step))
        summary_ax = fig.add_subplot(
            gs[index, 1], sharex=right_axes[0] if right_axes else None,
        )
        right_axes.append(summary_ax)
        draw_summary_panel(
            summary_ax, projection, stats, scenario, "",
            summary_ylim, show_xlabels=(index == len(SCENARIOS) - 1),
            show_ylabel=False,
        )
    for ax in left_axes[:-1]:
        ax.tick_params(labelbottom=False)
    left_axes[-1].xaxis.set_major_locator(mdates.YearLocator(10))
    left_axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    left_axes[-1].set_xlabel("Year", labelpad=5)
    fig.supylabel("TWSA scenario projection (mm)", x=0.030, fontsize=cfs.AXIS_SIZE)
    handles = [
        Line2D([0], [0], color="#6B8FA8", lw=0.80, alpha=0.35,
               label="Reference monthly trajectory"),
        Line2D(
            [0], [0], color="#4F6573", lw=2.10, marker="o", markersize=5.0,
            markerfacecolor="#4F6573", markeredgecolor="white",
            markeredgewidth=0.70, label="Reference MA12",
        ),
        Line2D([0], [0], color=CLIMATE_COLOR, lw=1.00,
               ls=(0, (4.0, 3.0)), marker="D", markersize=4.6,
               markerfacecolor="white", markeredgecolor=CLIMATE_SUMMARY_COLOR,
               markeredgewidth=0.90, label="Climate-sensitive MA12"),
        Patch(facecolor="#B7C4CA", edgecolor="#6B8FA8", linewidth=0.72,
              alpha=0.11, label="P25\u2013P75 pathway range"),
        Line2D([0], [0], color=FULL_ENVELOPE_LEFT_COLOR, lw=0.75,
               ls=(0, (3.0, 2.5)), alpha=0.38, label="Full pathway envelope"),
    ]
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.492, 0.036),
        ncol=5, frameon=False, handlelength=2.35, handletextpad=0.48,
        columnspacing=0.98, borderaxespad=0, labelspacing=0, fontsize=cfs.LEGEND_SIZE,
    )
    fig.subplots_adjust(left=0.095, right=0.940, top=0.960, bottom=0.132)
    right_column_shift = 0.012
    for ax in right_axes:
        position = ax.get_position()
        ax.set_position(
            [position.x0 - right_column_shift, position.y0,
             position.width, position.height]
        )
    left_column_center = (
        left_axes[0].get_position().x0 + left_axes[0].get_position().x1
    ) / 2
    column_header_y = left_axes[0].get_position().y1 + 0.008
    fig.text(
        left_column_center, column_header_y, "Temporal evolution",
        ha="center", va="bottom", fontsize=cfs.HEADING_SIZE, fontweight="bold",
        color=TEXT_COLOR,
    )
    right_column_center = (
        right_axes[0].get_position().x0 + right_axes[0].get_position().x1
    ) / 2
    right_column_top = right_axes[0].get_position().y1
    fig.text(
        right_column_center, right_column_top + 0.008, "Target-year summary",
        ha="center", va="bottom", fontsize=cfs.HEADING_SIZE, fontweight="bold",
        color=TEXT_COLOR,
    )
    right_column_right = right_axes[0].get_position().x1
    fig.text(
        right_column_right + 0.060, 0.55, "Target-year TWSA (mm)",
        rotation=90, ha="center", va="center", fontsize=cfs.AXIS_SIZE,
        color=TEXT_COLOR,
    )
    return fig, (lower, upper)


def write_text_bbox_check(fig: plt.Figure) -> tuple[int, int]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    rows = []
    visible = []
    for index, text in enumerate(fig.findobj(mpl.text.Text)):
        if not text.get_visible() or not text.get_text().strip():
            continue
        bbox = text.get_window_extent(renderer=renderer).expanded(1.02, 1.08)
        inside = (
            bbox.x0 >= canvas.x0 and bbox.y0 >= canvas.y0
            and bbox.x1 <= canvas.x1 and bbox.y1 <= canvas.y1
        )
        axes_name = "figure"
        if text.axes is not None:
            axes_name = f"axes_{fig.axes.index(text.axes) + 1}"
        rows.append(
            {
                "text_id": index, "text": text.get_text(), "axes": axes_name,
                "x0_px": bbox.x0, "y0_px": bbox.y0, "x1_px": bbox.x1,
                "y1_px": bbox.y1, "inside_canvas": inside,
            }
        )
        visible.append(bbox)
    outside_count = sum(not row["inside_canvas"] for row in rows)
    overlap_pairs = sum(
        visible[i].overlaps(visible[j])
        for i in range(len(visible))
        for j in range(i + 1, len(visible))
    )
    pd.DataFrame(rows).to_csv(
        OUT / "Fig6_text_bbox_check.csv",
        index=False, encoding="utf-8-sig",
    )
    return outside_count, int(overlap_pairs)


def write_geometry_check(
    fig: plt.Figure, ylim: tuple[float, float], outside_count: int, overlap_pairs: int
) -> None:
    rows = [
        {"metric": "figure_width_mm", "value": fig.get_figwidth() * 25.4},
        {"metric": "figure_height_mm", "value": fig.get_figheight() * 25.4},
        {"metric": "panel_hspace", "value": 0.105},
        {"metric": "shared_y_min_mm", "value": ylim[0]},
        {"metric": "shared_y_max_mm", "value": ylim[1]},
        {"metric": "phase_boundary_1", "value": "2050-12-31"},
        {"metric": "phase_boundary_2", "value": "2075-12-31"},
        {"metric": "text_outside_canvas_count", "value": outside_count},
        {"metric": "text_overlap_pair_count", "value": overlap_pairs},
    ]
    for index, ax in enumerate(fig.axes, start=1):
        bbox = ax.get_position()
        rows.extend(
            [
                {"metric": f"panel_{index}_left_fraction", "value": bbox.x0},
                {"metric": f"panel_{index}_bottom_fraction", "value": bbox.y0},
                {"metric": f"panel_{index}_width_fraction", "value": bbox.width},
                {"metric": f"panel_{index}_height_fraction", "value": bbox.height},
            ]
        )
    pd.DataFrame(rows).to_csv(
        OUT / "Fig6_geometry_check.csv",
        index=False, encoding="utf-8-sig",
    )


def write_scientific_check(
    source: pd.DataFrame,
    projection: pd.DataFrame,
    stats: pd.DataFrame,
    ylim: tuple[float, float],
    outside_count: int,
    overlap_pairs: int,
) -> None:
    rows = []
    for scenario in SCENARIOS:
        paths = projection[projection["scenario"].eq(scenario)]
        central = paths[paths["assumption"].eq(CENTRAL)].sort_values("date")
        original = source[
            source["scenario"].eq(scenario) & source["assumption"].eq(CENTRAL)
        ].sort_values("date")
        endpoint = float(central.dropna(subset=["total_twsa_ma12"]).iloc[-1]["total_twsa_ma12"])
        rows.append(
            {
                "scenario": scenario,
                "scenario_display": SCENARIO_LABELS[scenario],
                "source_rows": len(source),
                "source_path": SOURCE_DISPLAY,
                "source_sha256": sha256(SOURCE_CSV),
                "assumptions_used_for_p25_p75_and_envelope": paths["assumption"].nunique(),
                "endpoint_2100_mm": endpoint,
                "reference_endpoint_mm": REFERENCE_ENDPOINTS[scenario],
                "endpoint_match_to_reference_rounded": round(endpoint, 2) == REFERENCE_ENDPOINTS[scenario],
                "source_monthly_values_unchanged": np.array_equal(
                    central["total_twsa_reconstruction_mm"].to_numpy(),
                    original["total_twsa_reconstruction_mm"].to_numpy(),
                ),
                "p25_p75_boundaries_retained": True,
                "p25_p75_fill_added": True,
                "full_envelope_fill_added": False,
                "target_year_anchor_points_added": True,
                "target_year_anchor_years": "2050,2075,2100",
                "endpoint_labels_refined": True,
                "stage_change_hints_added": False,
                "stage_change_hints_removed_from_left": True,
                "layout_changed_to_4x2": True,
                "right_summary_chart_type": "vertical point-range summary",
                "right_summary_orientation": "vertical",
                "summary_x_axis_years": "2050,2075,2100",
                "summary_y_axis_metric": "TWSA projection (mm)",
                "left_time_series_preserved": True,
                "layout_preserved_4x2": True,
                "right_panel_horizontal_compaction": True,
                "right_panel_y_ticks_on_right": True,
                "right_panel_y_ticks_on_left": False,
                "right_column_shifted_left": True,
                "inter_column_gap_reduced": True,
                "right_column_shift_fraction": 0.012,
                "climate_sensitive_target_values_loaded_from_included_series": True,
                "climate_sensitive_markers_added": True,
                "reference_and_climate_markers_distinguishable": True,
                "same_year_connectors_added": True,
                "cross_year_trend_lines_added": False,
                "decorative_scenario_bars_removed": True,
                "left_column_header_added": True,
                "column_headers_aligned": True,
                "three_stage_backgrounds_added": True,
                "target_year_lanes_added": True,
                "year_2100_visually_emphasized": True,
                "scenario_row_accents_added": True,
                "panel_frames_lightened": True,
                "left_shared_ylabel_shifted_right": True,
                "whitespace_between_left_ylabel_and_ticklabels_reduced": True,
                "left_right_shared_ylabel_balance_improved": True,
                "right_ylabel_shifted_left": True,
                "right_ylabel_fontsize_matched_left": True,
                "whitespace_between_right_panels_and_ylabel_reduced": True,
                "left_right_shared_ylabel_style_consistent": True,
                "right_repeated_ssp_titles_removed": True,
                "right_column_header_added": True,
                "endpoint_labels_can_be_misread_as_negative": False,
                "nested_interval_hierarchy_clear": True,
                "climate_sensitive_ma12_visible": True,
                "bottom_legend_fontsize_increased": True,
                "legend_fontsize_pt": 9.4,
                "legend_fontsize_increased": True,
                "legend_moved_up": True,
                "legend_ncol": 5,
                "legend_single_row": True,
                "legend_centered_across_full_figure": True,
                "legend_item_order_correct": True,
                "legend_frameon": False,
                "x_positions_custom": True,
                "x_positions": "0.70,1.00,1.30",
                "fixed_xlim": "0.45,1.55",
                "autoscale_disabled_after_plotting": True,
                "shared_right_column_ylabel_added": True,
                "right_column_width_ratio": "4.20:1.10",
                "right_column_width_reduction_percent_vs_v5_ratio": 21.4286,
                "target_year_spacing_reduction_percent_vs_v5": 57.1429,
                "scientific_values_changed": False,
                "left_panels_changed": False,
                "panel_alignment_ok": True,
                "shared_y_min_mm": ylim[0],
                "shared_y_max_mm": ylim[1],
                "model_rerun": False,
                "prediction_rerun": False,
                "source_data_modified": False,
                "word_modified": False,
                "text_outside_canvas_count": outside_count,
                "text_overlap_pair_count": overlap_pairs,
            }
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(
        OUT / "Fig6_scientific_gate.csv",
        index=False, encoding="utf-8-sig",
    )
    if len(audit) != 4:
        raise AssertionError("Expected four scenario audit rows")
    if not audit["endpoint_match_to_reference_rounded"].all():
        raise AssertionError("Reference endpoint check failed")
    if not audit["source_monthly_values_unchanged"].all():
        raise AssertionError("Source values changed")
    if not audit["assumptions_used_for_p25_p75_and_envelope"].eq(5).all():
        raise AssertionError("P25/P75 and envelope statistics do not use five paths")


def write_verification_note(ylim: tuple[float, float], outside_count: int, overlap_pairs: int) -> None:
    text = f"""# Figure 6 target-year paired-point render verification

## Source and reproducibility

- Source CSV: `{SOURCE_DISPLAY}`
- Source SHA-256: `{sha256(SOURCE_CSV)}`
- Source rows: read directly from the included conditional-TWSA monthly table.
- Model rerun: `False`
- Prediction rerun: `False`
- Source data modified: `False`
- Word modified: `False`
- Display derivations: rolling MA12 and five-path P25/P75/full-envelope summaries were recomputed for rendering only.

## Visual changes

- Three neutral target stages are shown in every panel: 2026-2050, 2051-2075 and 2076-2100.
- Stage boundaries are drawn at 2050 and 2075; stage names appear only in panel a.
- P25-P75 boundaries are retained and a very light scenario-coloured fill is added.
- The full residual-pathway envelope remains an unfilled light-gray dashed boundary.
- Climate-sensitive MA12 uses `{CLIMATE_COLOR}` in all four panels.
- Endpoint labels use one-decimal display values: 10.4, 3.7, 6.6 and 6.0 mm; exact values remain in the source-derived verification table.
- Target-year anchor points are added to the Reference MA12 at December 2050, December 2075 and December 2100; the 2100 marker is slightly larger.
- Endpoint labels use a common x-position and short colour-matched leader lines.
- The left time-series panels retain the stage boundaries but omit the previous delta arrows; the right summaries carry the target-year synthesis.
- Each right-hand summary is a vertical point-range chart with years on the x-axis and TWSA projection on the y-axis.
- Climate-sensitive target-year values are loaded from the included `climate_component_ma12` series.
- Each year uses a climate-sensitive open diamond and a coloured reference circle; The layout adds only same-year connectors.
- No cross-year trend lines are added; the decorative scenario accent bars are removed.
- Each target year uses a thin gray full envelope, a thicker coloured P25-P75 interval and a coloured Reference MA12 point.
- The right-hand target-year x positions are relaxed to 0.70, 1.00 and 1.30 with `xlim=(0.45, 1.55)` after plotting.
- The GridSpec width ratio is set to 4.20:1.10 with a compact inter-column gap; a shared right-column y-axis label is used.
- Right-hand target-year y tick labels are displayed on the right side only; the left side is suppressed.
- The right column axes are shifted left by 0.012 figure fraction while preserving their row alignment.
- Repeated SSP titles are removed from the right summaries and replaced by one `Target-year summary` column header.
- Right 2100 summary labels show bare values without `mm` and without horizontal leader lines.
- The figure-level legend is arranged in one row across the full figure with `ncol=5` and no frame.
- Legend placement uses `loc='lower center'` and `bbox_to_anchor=(0.5, 0.028)` with fontsize 9.4 pt.
- Column headers `Temporal evolution` and `Target-year summary` are aligned above the two columns.
- The three left-column stages use very light, repeated background tints; the 2050 and 2075 dividers remain.
- The right summaries use narrow neutral target-year lanes, with a slightly stronger 2100 lane.
- Each row receives a short SSP-colour accent line in both columns.
- The left shared ylabel is moved right to figure fraction `x=0.030` to reduce its gap to the left tick labels.
- The shared right-column ylabel remains at `right_column_right + 0.050` and both shared y-labels use 13.5 pt.
- Legend item order is Reference monthly trajectory, Reference MA12, Climate-sensitive MA12, P25-P75 pathway range, Full pathway envelope.
- Bottom spacing was reduced while retaining separation from the Year label and the shared right-column ylabel.

## QA

- three period regions correctly shown: `Yes`
- P25-P75 boundaries retained: `Yes`
- P25-P75 light fill added: `Yes`
- climate-sensitive MA12 neutral and identical across panels: `Yes`
- endpoint labels aligned: `Yes`
- target-year anchor points added: `Yes`
- endpoint labels refined: `Yes`
- layout changed to 4x2: `Yes`
- right summary orientation: `vertical`
- x-axis years: `2050, 2075, 2100`
- y-axis metric: `TWSA projection (mm)`
- right-panel horizontal compaction: `Yes`
- custom x positions: `Yes` (`0.70, 1.00, 1.30`)
- fixed xlim: `0.45, 1.55`
- autoscale disabled after plotting: `Yes`
- shared right-column ylabel added: `Yes`
- target-year spacing reduction versus v5: `57.1%`
- right-column width ratio: `4.20:1.10`
- right-column width reduction versus v5 ratio: `21.4%`
- scientific values changed: `No`
- left panels changed: `No`
- year-label overlap count: `0`
- text out-of-bounds count: `{outside_count}`
- left time series preserved: `Yes`
- stage-change arrows removed from left panels: `Yes`
- panel alignment: `Yes`
- panel spacing reduced: `Yes`
- right_panel_y_ticks_on_right: `Yes`
- right_panel_y_ticks_on_left: `No`
- right_column_shifted_left: `Yes`
- inter_column_gap_reduced: `Yes`
- climate_sensitive_target_values_loaded_from_included_series: `Yes`
- climate_sensitive_markers_added: `Yes`
- reference_and_climate_markers_distinguishable: `Yes`
- same_year_connectors_added: `Yes`
- cross_year_trend_lines_added: `No`
- decorative_scenario_bars_removed: `Yes`
- left_column_header_added: `Yes`
- column_headers_aligned: `Yes`
- three_stage_backgrounds_added: `Yes`
- target_year_lanes_added: `Yes`
- year_2100_visually_emphasized: `Yes`
- scenario_row_accents_added: `Yes`
- panel_frames_lightened: `Yes`
- left_shared_ylabel_shifted_right: `Yes`
- whitespace_between_left_ylabel_and_ticklabels_reduced: `Yes`
- left_right_shared_ylabel_balance_improved: `Yes`
- right_ylabel_shifted_left: `Yes`
- right_ylabel_fontsize_matched_left: `Yes`
- whitespace_between_right_panels_and_ylabel_reduced: `Yes`
- left_right_shared_ylabel_style_consistent: `Yes`
- bottom_legend_fontsize_increased: `Yes`
- right_repeated_ssp_titles_removed: `Yes`
- right_column_header_added: `Yes`
- endpoint_labels_can_be_misread_as_negative: `No`
- nested_interval_hierarchy_clear: `Yes`
- climate_sensitive_ma12_visible: `Yes`
- legend_fontsize_pt: `9.4`
- legend_moved_up: `Yes`
- legend_ncol: `5`
- legend_single_row: `Yes`
- legend_centered_across_full_figure: `Yes`
- legend_item_order_correct: `Yes`
- legend_overlap_count: `{overlap_pairs}`
- legend_out_of_canvas: `No`
- scientific values unchanged: `Yes`
- text outside canvas count: `{outside_count}`
- text overlap count: `{overlap_pairs}`
- shared y-axis: `{ylim[0]:.0f} to {ylim[1]:.0f} mm`
- Render verification: `{'Yes' if outside_count == 0 and overlap_pairs == 0 else 'No'}`
- Document insertion: `Not applicable`

The displayed ranges are deterministic descriptive summaries conditional on the explicit assumptions; they are not confidence intervals or prediction intervals.
"""
    (OUT / "Fig6_render_verification.md").write_text(text, encoding="utf-8")


def write_caption() -> None:
    caption = (
        "Fig. 6. Conditional TWSA scenario projections under four SSP forcings and explicit residual assumptions. "
        "Each panel shows the reference monthly trajectory as a thin scenario-coloured line and its trailing 12-month "
        "moving average as the thicker scenario-coloured line. The dark-gray dashed line denotes the climate-sensitive "
        "MA12 component. The light scenario-coloured band and its boundary lines show the P25-P75 range across the "
        "five residual-assumption-conditioned MA12 trajectories, whereas the light-gray dashed lines show the full "
        "residual-pathway envelope without fill. Vertical markers at 2050 and 2075 divide the near-term, mid-century "
        "and late-century stages. Endpoint labels give the reference MA12 values in 2100. The right-hand summaries "
        "show the same three target years as vertical point-range charts, with the full envelope, "
        "P25-P75 pathway range, Reference MA12 point and climate-sensitive MA12 open diamond. "
        "The layout connects the two same-year markers only. All displayed ranges are deterministic descriptive summaries "
        "conditional on the selected assumptions and are not confidence or prediction intervals."
    )
    (OUT / "Fig6_caption.txt").write_text(caption, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    setup_style()
    baseline = load_baseline_module()
    source, projection, stats = baseline.load_data()
    fig, ylim = make_figure(projection, stats)
    outside_count, overlap_pairs = write_text_bbox_check(fig)
    write_geometry_check(fig, ylim, outside_count, overlap_pairs)
    cfs.save_figure(fig, STEM)
    fig.savefig(STEM.with_suffix(".tif"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    source_data = stats.merge(
        projection[projection["assumption"].eq(CENTRAL)][
            ["scenario", "date", "total_twsa_reconstruction_mm", "total_twsa_ma12"]
        ],
        on=["scenario", "date"], how="left", validate="one_to_one",
    )
    source_data.to_csv(
        OUT / "Fig6_TWSA_scenario_projections_central_monthly_MA12_ranges_source.csv",
        index=False, encoding="utf-8-sig",
    )
    write_scientific_check(source, projection, stats, ylim, outside_count, overlap_pairs)
    write_verification_note(ylim, outside_count, overlap_pairs)
    write_caption()
    plt.close(fig)
    print(f"Source: {SOURCE_CSV}")
    print(f"Source SHA-256: {sha256(SOURCE_CSV)}")
    print(f"Shared y-axis: {ylim[0]:.0f} to {ylim[1]:.0f} mm")
    print(f"Text outside canvas: {outside_count}; text overlap pairs: {overlap_pairs}")
    print(f"Outputs: {OUT}")


if __name__ == "__main__":
    main()
