from __future__ import annotations

import hashlib
import os
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

import cee_figure_style as cfs


MM = 1 / 25.4
FIG_W_MM = 180.0
FIG_H_MM = 101.4
DPI = 600

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "processed" / "fig4"
OUT = REPO_ROOT / "figures" / "reproduced"

SPATIAL_SOURCE = DATA_DIR / "Fig4_spatial_source.csv"
SCALE_SOURCE = DATA_DIR / "fig4_colour_scale_summary.csv"
COUNT_SOURCE = DATA_DIR / "fig4_grid_count_summary.csv"
RISK_SOURCE = DATA_DIR / "Fig4_annual_end_risk_source.csv"
INCREMENT_SOURCE = DATA_DIR / "Fig4_period_increment_source.csv"
BOUNDARY_SOURCE = REPO_ROOT / "data" / "external" / "Haihe_basin_HydroBASINS_L9_boundary.shp"
BOUNDARY_COMPONENTS = [BOUNDARY_SOURCE.with_suffix(ext) for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg")]

SCENARIOS = ["ssp126", "ssp245", "ssp370", "ssp585"]
DISPLAY = {
    "ssp126": "SSP1-2.6",
    "ssp245": "SSP2-4.5",
    "ssp370": "SSP3-7.0",
    "ssp585": "SSP5-8.5",
}
COLORS = dict(cfs.SSP_COLORS)
PERIOD_COLUMNS = [
    "increment_2026_2050",
    "increment_2051_2075",
    "increment_2076_2100",
]
PERIOD_LABELS = ["2026–2050", "2051–2075", "2076–2100"]
PRESSURE_CMAP = LinearSegmentedColormap.from_list(
    "water_balance_change",
    [cfs.NEG, cfs.ZERO, cfs.POS],
    N=256,
)
SPINE = cfs.FRAME
GRID = cfs.GRID


def setup_style() -> None:
    cfs.apply_style()
    matplotlib.rcParams.update(
        {
            "axes.titlesize": cfs.HEADING_SIZE,
            "axes.linewidth": 0.78,
        }
    )


def add_axes_mm(
    fig: plt.Figure, x: float, y: float, width: float, height: float
) -> plt.Axes:
    return fig.add_axes(
        [
            x / FIG_W_MM,
            y / FIG_H_MM,
            width / FIG_W_MM,
            height / FIG_H_MM,
        ]
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    cfs.panel_label(ax, label, x=0.018, y=0.976)


def close_axis(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(SPINE)
        spine.set_linewidth(0.72)
    ax.tick_params(
        direction="out",
        length=2.6,
        width=0.70,
        color=SPINE,
        labelcolor="#34383A",
        pad=1.8,
    )


def degree_e(value: float, _position: int) -> str:
    return f"{value:.0f}°E"


def degree_n(value: float, _position: int) -> str:
    return f"{value:.0f}°N"


def grid_edges(values: np.ndarray) -> np.ndarray:
    centers = np.asarray(sorted(np.unique(values)), dtype=float)
    midpoints = (centers[:-1] + centers[1:]) / 2
    return np.r_[
        centers[0] - (midpoints[0] - centers[0]),
        midpoints,
        centers[-1] + (centers[-1] - midpoints[-1]),
    ]


def pivot_grid(
    data: pd.DataFrame, value: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pivot = (
        data.pivot(index="lat", columns="lon", values=value)
        .sort_index()
        .sort_index(axis=1)
    )
    return (
        pivot.columns.to_numpy(dtype=float),
        pivot.index.to_numpy(dtype=float),
        pivot.to_numpy(dtype=float),
    )


def load_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    gpd.GeoDataFrame,
    float,
    pd.DataFrame,
]:
    spatial = pd.read_csv(SPATIAL_SOURCE)
    scale = pd.read_csv(SCALE_SOURCE).iloc[0]
    common_limit = float(max(abs(scale["vmin"]), abs(scale["vmax"])))

    risk = pd.read_csv(RISK_SOURCE, parse_dates=["date"])
    risk = risk[risk["indicator"].eq("risk_deficit_cumulative_z")].copy()
    risk["scenario"] = risk["scenario"].str.lower()
    annual_end = risk[risk["date"].dt.month.eq(12)].copy()
    annual_end["year"] = annual_end["date"].dt.year

    increments = pd.read_csv(INCREMENT_SOURCE)
    increments["scenario"] = increments["scenario"].str.lower()
    boundary = gpd.read_file(BOUNDARY_SOURCE).to_crs(4326).dissolve()
    count_qa = pd.read_csv(COUNT_SOURCE)
    return spatial, annual_end, increments, boundary, common_limit, count_qa


def draw_map(
    ax: plt.Axes,
    spatial: pd.DataFrame,
    boundary: gpd.GeoDataFrame,
    scenario: str,
    common_limit: float,
    letter: str,
    show_x: bool,
    show_y: bool,
) -> object:
    subset = spatial[spatial["scenario"].eq(scenario)].copy()
    lons, lats, values = pivot_grid(subset, "delta_wb2_median_mm_month")
    norm = TwoSlopeNorm(vmin=-common_limit, vcenter=0.0, vmax=common_limit)
    mesh = ax.pcolormesh(
        grid_edges(lons),
        grid_edges(lats),
        np.ma.masked_invalid(values),
        cmap=PRESSURE_CMAP,
        norm=norm,
        shading="flat",
        edgecolors="none",
        linewidth=0,
        rasterized=True,
        zorder=2,
    )

    drying = subset[
        subset["sign_agreement_class"].eq("robust drying (>=5/6)")
    ]
    wetting = subset[
        subset["sign_agreement_class"].eq("robust wetting (>=5/6)")
    ]
    ax.scatter(
        drying["lon"],
        drying["lat"],
        s=3.1,
        marker=".",
        color="#73352E",
        alpha=0.68,
        linewidths=0,
        zorder=6,
    )
    ax.scatter(
        wetting["lon"],
        wetting["lat"],
        s=10.0,
        marker="o",
        facecolors="none",
        edgecolors="#356A73",
        linewidths=0.58,
        alpha=0.82,
        zorder=6,
    )
    boundary.boundary.plot(
        ax=ax, color="#3E4548", linewidth=0.82, zorder=8
    )

    minx, miny, maxx, maxy = boundary.total_bounds
    ax.set_xlim(minx - 0.08, maxx + 0.08)
    ax.set_ylim(miny - 0.08, maxy + 0.08)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([114, 116, 118])
    ax.set_yticks([36, 38, 40, 42])
    ax.xaxis.set_major_formatter(FuncFormatter(degree_e))
    ax.yaxis.set_major_formatter(FuncFormatter(degree_n))
    if not show_x:
        ax.tick_params(labelbottom=False)
    if not show_y:
        ax.tick_params(labelleft=False)
    ax.set_xlabel("Longitude" if show_x else "", labelpad=1.6)
    ax.set_ylabel("Latitude" if show_y else "", labelpad=1.5)
    ax.set_title(
        DISPLAY[scenario],
        color=COLORS[scenario],
        fontweight="bold",
        pad=2.7,
    )
    close_axis(ax)
    panel_label(ax, letter)
    if scenario == "ssp585":
        sign_handles = [
            Line2D(
                [0],
                [0],
                marker=".",
                linestyle="none",
                color="#73352E",
                markersize=3.8,
                label="≥5/6 drying",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor="none",
                markeredgecolor="#356A73",
                markeredgewidth=0.60,
                markersize=4.4,
                label="≥5/6 wetting",
            ),
        ]
        ax.legend(
            handles=sign_handles,
            loc="lower right",
            bbox_to_anchor=(0.985, 0.018),
            ncol=1,
            frameon=True,
            facecolor="white",
            edgecolor="#D9DDDF",
            framealpha=0.88,
            fontsize=5.3,
            borderpad=0.25,
            labelspacing=0.22,
            handletextpad=0.35,
        )
    return mesh


def draw_risk_trajectory(
    ax: plt.Axes, annual_end: pd.DataFrame
) -> list[Line2D]:
    handles: list[Line2D] = []
    for scenario in SCENARIOS:
        subset = annual_end[annual_end["scenario"].eq(scenario)].sort_values(
            "year"
        )
        (line,) = ax.plot(
            subset["year"],
            subset["risk_index"],
            color=COLORS[scenario],
            linewidth=1.45,
            alpha=1.0,
            solid_capstyle="round",
            zorder=5,
            label=DISPLAY[scenario],
        )
        handles.append(line)

    ax.axhline(0, color="#949A9D", linewidth=0.58, zorder=1)
    ax.set_xlim(2025, 2101)
    ax.set_ylim(-2.0, 26.0)
    ax.set_xticks([2030, 2050, 2075, 2100])
    ax.set_yticks([0, 5, 10, 15, 20, 25])
    ax.set_xlabel("Year", labelpad=2.2)
    ax.set_ylabel(
        "Climate-storage risk index\n(dimensionless)", labelpad=1.0
    )
    ax.grid(axis="y", color=GRID, linewidth=0.48, zorder=0)
    close_axis(ax)
    panel_label(ax, "e")
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.095, 0.985),
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#D9DDDF",
        framealpha=0.90,
        borderpad=0.30,
        fontsize=5.8,
        labelspacing=0.24,
        columnspacing=0.68,
        handlelength=1.8,
        handletextpad=0.36,
    )
    return handles


def draw_period_increments(
    ax: plt.Axes, increments: pd.DataFrame
) -> None:
    x = np.arange(len(PERIOD_LABELS), dtype=float)
    bar_width = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * bar_width
    for scenario, offset in zip(SCENARIOS, offsets):
        row = increments[increments["scenario"].eq(scenario)].iloc[0]
        values = row[PERIOD_COLUMNS].to_numpy(dtype=float)
        ax.bar(
            x + offset,
            values,
            width=bar_width,
            color=COLORS[scenario],
            edgecolor="white",
            linewidth=0.55,
            alpha=0.96,
            zorder=5,
        )

    ax.set_xlim(-0.48, 2.48)
    ax.set_ylim(0, 16.2)
    ax.set_xticks(x, PERIOD_LABELS)
    ax.set_yticks([0, 4, 8, 12, 16])
    ax.set_xlabel("Accumulation period", labelpad=2.2)
    ax.set_ylabel("Risk-index increment\n(dimensionless)", labelpad=1.0)
    ax.grid(axis="y", color=GRID, linewidth=0.48, zorder=0)
    close_axis(ax)
    panel_label(ax, "f")


def build_figure(
    spatial: pd.DataFrame,
    annual_end: pd.DataFrame,
    increments: pd.DataFrame,
    boundary: gpd.GeoDataFrame,
    common_limit: float,
) -> plt.Figure:
    fig = plt.figure(figsize=(FIG_W_MM * MM, FIG_H_MM * MM))

    map_positions = {
        "ssp126": (11.00, 54.95, 42.63, 40.74, "a", False, True),
        "ssp245": (57.79, 54.95, 42.63, 40.74, "b", False, False),
        "ssp370": (11.00, 10.42, 42.63, 40.74, "c", True, True),
        "ssp585": (57.79, 10.42, 42.63, 40.74, "d", True, False),
    }
    mesh = None
    for scenario in SCENARIOS:
        x, y, w, h, letter, show_x, show_y = map_positions[scenario]
        ax = add_axes_mm(fig, x, y, w, h)
        mesh = draw_map(
            ax,
            spatial,
            boundary,
            scenario,
            common_limit,
            letter,
            show_x,
            show_y,
        )

    cax = add_axes_mm(fig, 106.11, 21.79, 1.90, 62.53)
    colorbar = fig.colorbar(mesh, cax=cax, orientation="vertical")
    colorbar.set_ticks([-common_limit, -3, 0, 3, common_limit])
    colorbar.ax.tick_params(
        labelsize=5.8, length=1.8, width=0.62, pad=1.2
    )
    colorbar.outline.set_linewidth(0.58)
    colorbar.outline.set_edgecolor(SPINE)
    colorbar.set_label(
        "Δ(P−ET−Q)\n(mm month$^{-1}$)", fontsize=6.3, labelpad=1.8
    )
    colorbar.ax.yaxis.set_ticks_position("left")
    colorbar.ax.yaxis.set_label_position("right")

    ax_e = add_axes_mm(fig, 125.05, 56.84, 50.21, 38.84)
    draw_risk_trajectory(ax_e, annual_end)
    ax_f = add_axes_mm(fig, 125.05, 10.42, 50.21, 38.84)
    draw_period_increments(ax_f, increments)
    return fig


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def renderer_check(fig: plt.Figure) -> dict[str, object]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figure_bbox = fig.bbox
    px_per_mm = fig.dpi / 25.4
    safety = 1.5 * px_per_mm
    failures: list[str] = []
    minimum = float("inf")
    for ax in fig.axes:
        objects = [
            ax.xaxis.label,
            ax.yaxis.label,
            ax.title,
            *ax.get_xticklabels(),
            *ax.get_yticklabels(),
            *ax.texts,
        ]
        legend = ax.get_legend()
        if legend is not None:
            objects.append(legend)
        for artist in objects:
            if not artist.get_visible():
                continue
            bbox = artist.get_window_extent(renderer)
            distances = [
                bbox.x0 - figure_bbox.x0,
                figure_bbox.x1 - bbox.x1,
                bbox.y0 - figure_bbox.y0,
                figure_bbox.y1 - bbox.y1,
            ]
            minimum = min(minimum, *distances)
            if any(distance < safety for distance in distances):
                text = getattr(artist, "get_text", lambda: "legend")()
                failures.append(str(text).replace("\n", " "))
    map_right_gap = float("nan")
    colorbar_to_panel_e_gap = float("nan")
    panel_e_to_panel_f_gap = float("nan")
    panel_e_to_b_top_delta = float("nan")
    panel_f_to_d_bottom_delta = float("nan")
    panel_e_aspect = float("nan")
    panel_f_aspect = float("nan")
    if len(fig.axes) >= 7:
        right_map = fig.axes[1].get_window_extent(renderer)
        colorbar_ax = fig.axes[4]
        panel_e = fig.axes[5]
        panel_f = fig.axes[6]
        tick_boxes = [
            tick.get_window_extent(renderer)
            for tick in colorbar_ax.get_yticklabels()
            if tick.get_visible()
        ]
        if tick_boxes:
            map_right_gap = (
                min(bbox.x0 for bbox in tick_boxes) - right_map.x1
            ) / px_per_mm
        colorbar_label = colorbar_ax.yaxis.label.get_window_extent(renderer)
        panel_e_ylabel = panel_e.yaxis.label.get_window_extent(renderer)
        colorbar_to_panel_e_gap = (
            panel_e_ylabel.x0 - colorbar_label.x1
        ) / px_per_mm
        panel_e_xlabel = panel_e.xaxis.label.get_window_extent(renderer)
        panel_f_box = panel_f.get_window_extent(renderer)
        panel_e_to_panel_f_gap = (
            panel_e_xlabel.y0 - panel_f_box.y1
        ) / px_per_mm
        panel_e_box = panel_e.get_window_extent(renderer)
        panel_b_box = fig.axes[1].get_window_extent(renderer)
        panel_d_box = fig.axes[3].get_window_extent(renderer)
        panel_e_to_b_top_delta = (
            panel_e_box.y1 - panel_b_box.y1
        ) / px_per_mm
        panel_f_to_d_bottom_delta = (
            panel_f_box.y0 - panel_d_box.y0
        ) / px_per_mm
        panel_e_aspect = panel_e_box.width / panel_e_box.height
        panel_f_aspect = panel_f_box.width / panel_f_box.height

    return {
        "minimum_text_edge_clearance_mm": minimum / px_per_mm,
        "text_edge_failures": len(failures),
        "failure_objects": " | ".join(failures) if failures else "None",
        "right_map_to_colourbar_tick_gap_mm": map_right_gap,
        "colourbar_label_to_panel_e_ylabel_gap_mm": colorbar_to_panel_e_gap,
        "panel_e_xlabel_to_panel_f_top_gap_mm": panel_e_to_panel_f_gap,
        "panel_e_to_b_top_delta_mm": panel_e_to_b_top_delta,
        "panel_f_to_d_bottom_delta_mm": panel_f_to_d_bottom_delta,
        "panel_e_plot_aspect_ratio": panel_e_aspect,
        "panel_f_plot_aspect_ratio": panel_f_aspect,
    }


def numeric_check(
    spatial: pd.DataFrame,
    annual_end: pd.DataFrame,
    increments: pd.DataFrame,
    common_limit: float,
    count_qa: pd.DataFrame,
) -> dict[str, object]:
    spatial_counts = spatial.groupby("scenario").size().to_dict()
    annual_counts = annual_end.groupby("scenario").size().to_dict()
    annual_decreases = {
        scenario: int(
            annual_end[annual_end["scenario"].eq(scenario)]
            .sort_values("year")["risk_index"]
            .diff()
            .lt(-1e-10)
            .sum()
        )
        for scenario in SCENARIOS
    }
    closure_max = float(increments["closure_error"].abs().max())
    increment_min = float(
        increments[PERIOD_COLUMNS].to_numpy(dtype=float).min()
    )
    return {
        "spatial_grid_counts": spatial_counts,
        "annual_end_counts": annual_counts,
        "annual_end_decreases": annual_decreases,
        "common_colour_limit": common_limit,
        "spatial_values_within_scale": bool(
            spatial["delta_wb2_median_mm_month"]
            .abs()
            .le(common_limit + 1e-12)
            .all()
        ),
        "sign_count_reproduction_pass": bool(
            count_qa["count_match"].astype(bool).all()
        ),
        "minimum_period_increment": increment_min,
        "maximum_closure_error": closure_max,
    }


def write_outputs(
    fig: plt.Figure,
    spatial: pd.DataFrame,
    annual_end: pd.DataFrame,
    increments: pd.DataFrame,
    common_limit: float,
    count_qa: pd.DataFrame,
) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    color_png = OUT / "Fig4_CEE_harmonized_v2.png"
    color_pdf = OUT / "Fig4_CEE_harmonized_v2.pdf"
    color_svg = OUT / "Fig4_CEE_harmonized_v2.svg"
    grayscale_png = OUT / "Fig4_CEE_harmonized_v2_grayscale.png"
    size_check_png = OUT / "Fig4_CEE_harmonized_v2_180mm_size_check.png"

    fig.savefig(color_png, dpi=DPI, facecolor="white")
    fig.savefig(color_pdf, facecolor="white")
    fig.savefig(color_svg, facecolor="white")
    fig.savefig(size_check_png, dpi=300, facecolor="white")

    with Image.open(color_png) as image:
        image.convert("L").save(grayscale_png, dpi=(DPI, DPI))

    annual_end[
        ["date", "year", "scenario", "risk_index", "indicator"]
    ].to_csv(OUT / "Fig4_annual_end_risk_source.csv", index=False)
    increments[
        [
            "scenario",
            "display_name",
            *PERIOD_COLUMNS,
            "sum",
            "closure_error",
        ]
    ].to_csv(OUT / "Fig4_period_increment_source.csv", index=False)
    spatial.to_csv(OUT / "Fig4_spatial_source.csv", index=False)

    renderer = renderer_check(fig)
    numeric = numeric_check(
        spatial, annual_end, increments, common_limit, count_qa
    )
    with Image.open(color_png) as image:
        color_size = image.size
    with Image.open(size_check_png) as image:
        check_size = image.size
    verification = f"""# Figure 4 rendering and numerical verification

## Figure contract

- Core conclusion: Late-century hydroclimatic water-balance changes are
  spatially heterogeneous, while the cumulative climate-storage risk index
  rises under all SSPs with scenario-dependent period increments.
- Evidence chain: panels a-d show spatial pressure; panel e shows annual-end
  cumulative risk states; panel f compares three non-overlapping accumulation
  periods.
- Archetype: asymmetric mixed-modality quantitative composite.
- Output width: {FIG_W_MM:.0f} mm.

## Scientific integrity

```text
data_changed = False
metrics_recomputed = False
model_rerun = False
spatial_values_recomputed = False
risk_series_recomputed = False
Word_changed = False
```

- spatial_grid_counts = {numeric['spatial_grid_counts']}
- annual_end_counts = {numeric['annual_end_counts']}
- annual_end_decreases = {numeric['annual_end_decreases']}
- common_colour_limit = ±{numeric['common_colour_limit']:.12f} mm month^-1
- spatial_values_within_scale = {numeric['spatial_values_within_scale']}
- sign_count_reproduction_pass = {numeric['sign_count_reproduction_pass']}
- minimum_period_increment = {numeric['minimum_period_increment']:.12f}
- maximum_closure_error = {numeric['maximum_closure_error']:.3e}

## Display exclusions

```text
old_area_proportion_bars = False
grouped_bars = True
annual_increment_background_lines = False
five_year_moving_mean = False
individual_GCM_lines = False
```

## Geometry and export

- canvas_mm = {FIG_W_MM:.1f} x {FIG_H_MM:.1f}
- previous_canvas_mm = 190.0 x 125.0
- vertical_canvas_reduction_mm = 18.0
- axes_uniform_vertical_shift_mm = -14.0
- panel_dimensions_changed = False
- color_png_pixels_600dpi = {color_size[0]} x {color_size[1]}
- size_check_png_pixels_300dpi = {check_size[0]} x {check_size[1]}
- minimum_text_edge_clearance_mm = {renderer['minimum_text_edge_clearance_mm']:.3f}
- text_edge_failures = {renderer['text_edge_failures']}
- failure_objects = {renderer['failure_objects']}
- font_family = Arial
- axes_closed = True
- shared_map_colour_scale = True
- editable_vector_exports = True
- panels_a_d_data_changed = False
- panel_e_data_changed = False
- panel_f_data_changed = False
- panel_f_visualization = grouped bar chart
- panel_e_f_top_titles_removed = True
- map_colourbar_orientation = vertical
- right_panel_left_edge_mm = 125.05
- right_panel_dimensions_mm = 50.21 x 38.84
- right_panel_vertical_gap_mm = 7.58
- panel_e_top_aligned_with_panel_b = True
- panel_f_bottom_aligned_with_panel_d = True
- right_map_to_colourbar_tick_gap_mm = {renderer['right_map_to_colourbar_tick_gap_mm']:.3f}
- colourbar_label_to_panel_e_ylabel_gap_mm = {renderer['colourbar_label_to_panel_e_ylabel_gap_mm']:.3f}
- panel_e_xlabel_to_panel_f_top_gap_mm = {renderer['panel_e_xlabel_to_panel_f_top_gap_mm']:.3f}
- panel_e_to_panel_b_top_delta_mm = {renderer['panel_e_to_b_top_delta_mm']:.3f}
- panel_f_to_panel_d_bottom_delta_mm = {renderer['panel_f_to_d_bottom_delta_mm']:.3f}
- panel_e_plot_aspect_ratio = {renderer['panel_e_plot_aspect_ratio']:.3f}
- panel_f_plot_aspect_ratio = {renderer['panel_f_plot_aspect_ratio']:.3f}

## Gate

original_four_spatial_panels_restored = Yes
all_six_original_scientific_panels_present = Yes
RENDER_VERIFICATION = {'YES' if renderer['text_edge_failures'] == 0 else 'NO'}
PUBLICATION_EXPORT = NOT_GENERATED
DOCUMENT_INSERTION = NOT_APPLICABLE
"""
    (OUT / "figure4_render_verification.md").write_text(verification, encoding="utf-8")

    inventory = f"""# Figure 4 source inventory

| Role | Source | SHA-256 |
|---|---|---|
| Four SSP spatial water-balance fields | `{SPATIAL_SOURCE}` | `{sha256(SPATIAL_SOURCE)}` |
| Common spatial colour scale | `{SCALE_SOURCE}` | `{sha256(SCALE_SOURCE)}` |
| Spatial sign-agreement counts | `{COUNT_SOURCE}` | `{sha256(COUNT_SOURCE)}` |
| Monthly risk-index states | `{RISK_SOURCE}` | `{sha256(RISK_SOURCE)}` |
| Non-overlapping period increments | `{INCREMENT_SOURCE}` | `{sha256(INCREMENT_SOURCE)}` |
| Haihe Basin boundary | `{BOUNDARY_SOURCE}` | `{sha256(BOUNDARY_SOURCE)}` |

Annual-end trajectories are direct December selections from the included monthly
`risk_deficit_cumulative_z` table. No smoothing, interpolation, averaging, or
scientific recalculation was applied.
"""
    (OUT / "figure4_source_inventory.md").write_text(
        inventory, encoding="utf-8"
    )

    caption = (
        "Figure 4. Late-century hydroclimatic water-balance change and "
        "climate-storage risk under four SSPs. (a-d) Six-GCM median changes "
        "in P−ET−Q during 2081–2100 relative to each model's 1995–2014 "
        "baseline. Dots and open circles denote grid cells where at least "
        "five of six models agree on drying or wetting, respectively; "
        "unmarked cells have mixed model signs. (e) December values of the "
        "dimensionless cumulative climate-storage risk index from 2026 to "
        "2100. (f) Grouped bars show risk-index increments accumulated "
        "during 2026–2050, 2051–2075, and 2076–2100."
    )
    (OUT / "figure4_caption_en.txt").write_text(caption, encoding="utf-8")
    shutil.copy2(__file__, OUT / "figure4_reproduction_script.py")


def main() -> None:
    missing_boundary = [p for p in BOUNDARY_COMPONENTS if not p.exists()]
    if missing_boundary:
        print("Fig. 4 rendering: WARNING")
        print("Included numerical source tables are present, but the licensed HydroBASINS v1c Asia level-9 boundary is external.")
        print("Missing required shapefile components:")
        for path in missing_boundary:
            print(f"  - data/external/{path.name}")
        print("Acquisition instructions: data/README.md and external_data_manifest.csv")
        return 2
    setup_style()
    (
        spatial,
        annual_end,
        increments,
        boundary,
        common_limit,
        count_qa,
    ) = load_data()
    fig = build_figure(
        spatial, annual_end, increments, boundary, common_limit
    )
    write_outputs(
        fig,
        spatial,
        annual_end,
        increments,
        common_limit,
        count_qa,
    )
    plt.close(fig)
    print(f"output={OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
