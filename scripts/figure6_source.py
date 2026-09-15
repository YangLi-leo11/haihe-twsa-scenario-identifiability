from __future__ import annotations

import hashlib
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = REPO_ROOT / "data" / "processed" / "fig5_fig6" / "conditional_twsa_monthly.csv"
OUT = REPO_ROOT / "figures" / "reproduced"
STEM = OUT / "Fig6_TWSA_scenario_projections_central_monthly_MA12_ranges"

SCENARIOS = ["ssp126", "ssp245", "ssp370", "ssp585"]
ASSUMPTIONS = [
    "A_stationary_residual",
    "B_trend_capped_persistence_residual",
    "C_risk_conditioned_residual",
    "D_moderation_residual",
    "E_consistency_reference_residual",
]
CENTRAL = "A_stationary_residual"
SCENARIO_LABELS = {
    "ssp126": "SSP1-2.6",
    "ssp245": "SSP2-4.5",
    "ssp370": "SSP3-7.0",
    "ssp585": "SSP5-8.5",
}
SCENARIO_COLORS = {
    "ssp126": "#2F6FA3",
    "ssp245": "#2E7D62",
    "ssp370": "#C58A18",
    "ssp585": "#9A4F83",
}
REFERENCE_ENDPOINTS = {
    "ssp126": 10.43,
    "ssp245": 3.67,
    "ssp370": 6.58,
    "ssp585": 6.02,
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 11.5,
            "axes.labelsize": 13.5,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10.3,
            "axes.linewidth": 0.85,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(SOURCE_CSV, parse_dates=["date"]).sort_values(
        ["scenario", "assumption", "date"]
    )
    required = {
        "date",
        "scenario",
        "assumption",
        "total_twsa_reconstruction_mm",
        "climate_component_mm",
    }
    if missing := required.difference(source.columns):
        raise ValueError(f"Missing source fields: {sorted(missing)}")
    if source["scenario"].drop_duplicates().tolist() != SCENARIOS:
        raise ValueError("Scenario order differs from the reference order")
    if source["assumption"].drop_duplicates().tolist() != ASSUMPTIONS:
        raise ValueError("Residual-assumption set differs from A-E")

    projection = source.copy()
    projection["total_twsa_ma12"] = projection.groupby(
        ["scenario", "assumption"], sort=False
    )["total_twsa_reconstruction_mm"].transform(
        lambda values: values.rolling(window=12, min_periods=6).mean()
    )

    # First compute MA12 within each scenario-assumption path. Boundary statistics
    # are then formed from those five low-frequency paths, not raw monthly values.
    stats = (
        projection.groupby(["scenario", "date"], sort=False)[
            "total_twsa_ma12"
        ]
        .agg(
            envelope_min_ma12="min",
            p25_ma12=lambda values: values.quantile(0.25),
            p75_ma12=lambda values: values.quantile(0.75),
            envelope_max_ma12="max",
        )
        .reset_index()
    )

    climate = (
        projection[["scenario", "date", "climate_component_mm"]]
        .drop_duplicates(["scenario", "date"])
        .sort_values(["scenario", "date"])
    )
    if len(climate) != len(SCENARIOS) * 900:
        raise ValueError("Climate component is not unique within scenario-date")
    climate["climate_component_ma12"] = climate.groupby("scenario", sort=False)[
        "climate_component_mm"
    ].transform(lambda values: values.rolling(window=12, min_periods=6).mean())
    stats = stats.merge(
        climate[["scenario", "date", "climate_component_ma12"]],
        on=["scenario", "date"],
        how="left",
        validate="one_to_one",
    )
    return source, projection, stats


def shared_limits(
    projection: pd.DataFrame, stats: pd.DataFrame
) -> tuple[float, float, float]:
    central_monthly = projection.loc[
        projection["assumption"].eq(CENTRAL), "total_twsa_reconstruction_mm"
    ]
    displayed_ma12 = pd.concat(
        [
            projection.loc[
                projection["assumption"].eq(CENTRAL), "total_twsa_ma12"
            ],
            stats["p25_ma12"],
            stats["p75_ma12"],
            stats["envelope_min_ma12"],
            stats["envelope_max_ma12"],
            stats["climate_component_ma12"],
        ],
        ignore_index=True,
    )
    values = pd.concat([central_monthly, displayed_ma12], ignore_index=True).dropna()
    minimum, maximum = float(values.min()), float(values.max())
    span = maximum - minimum
    padded_min = minimum - 0.05 * span
    padded_max = maximum + 0.05 * span
    # Tight display limits retain the requested 5% margin without forcing a
    # visually wasteful 100-mm rounding buffer. Labels remain at 100-mm steps.
    boundary_step = 20.0
    lower = math.floor(padded_min / boundary_step) * boundary_step
    upper = math.ceil(padded_max / boundary_step) * boundary_step
    if lower < -150:
        lower = -200.0
    return lower, upper, 100.0


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#E3E6E8", linewidth=0.65, zorder=0)
    ax.grid(axis="x", visible=False)
    ax.axhline(0, color="#92999D", linewidth=0.9, zorder=1)
    for spine in ax.spines.values():
        spine.set_color("#687176")
        spine.set_linewidth(0.85)
    ax.tick_params(
        direction="out",
        length=4,
        width=0.8,
        color="#687176",
        labelcolor="#30363A",
        pad=3.5,
    )


def draw_panel(
    ax: plt.Axes,
    projection: pd.DataFrame,
    stats: pd.DataFrame,
    scenario: str,
    panel_label: str,
    ylim: tuple[float, float],
) -> None:
    color = SCENARIO_COLORS[scenario]
    scenario_paths = projection[projection["scenario"].eq(scenario)]
    central = scenario_paths[scenario_paths["assumption"].eq(CENTRAL)].sort_values("date")
    bounds = stats[stats["scenario"].eq(scenario)].sort_values("date")

    ax.plot(
        bounds["date"], bounds["envelope_min_ma12"],
        color="#AEB4B8", linewidth=0.75, linestyle=(0, (3.0, 2.5)),
        alpha=0.60, zorder=3,
    )
    ax.plot(
        bounds["date"], bounds["envelope_max_ma12"],
        color="#AEB4B8", linewidth=0.75, linestyle=(0, (3.0, 2.5)),
        alpha=0.60, zorder=3,
    )
    ax.plot(
        central["date"], central["total_twsa_reconstruction_mm"],
        color=color, linewidth=0.68, alpha=0.33,
        solid_capstyle="round", solid_joinstyle="round",
        rasterized=True, zorder=4,
    )
    ax.plot(
        bounds["date"], bounds["p25_ma12"],
        color=color, linewidth=0.90, alpha=0.60, zorder=5,
    )
    ax.plot(
        bounds["date"], bounds["p75_ma12"],
        color=color, linewidth=0.90, alpha=0.60, zorder=5,
    )
    ax.plot(
        central["date"], central["total_twsa_ma12"],
        color=color, linewidth=2.6, alpha=1,
        solid_capstyle="round", solid_joinstyle="round", zorder=8,
    )
    ax.plot(
        bounds["date"], bounds["climate_component_ma12"],
        color="#363636", linewidth=1.15, linestyle=(0, (3.5, 2.4)),
        alpha=0.95, zorder=9,
    )

    endpoint = central.dropna(subset=["total_twsa_ma12"]).iloc[-1]
    endpoint_value = float(endpoint["total_twsa_ma12"])
    ax.scatter(
        endpoint["date"], endpoint_value, s=28, color=color,
        edgecolor="white", linewidth=0.8, zorder=10, clip_on=False,
    )
    ax.annotate(
        f"{endpoint_value:.2f} mm",
        xy=(endpoint["date"], endpoint_value), xytext=(-7, 10),
        textcoords="offset points", ha="right", va="bottom",
        fontsize=10, fontweight="bold", color=color,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.84, pad=1.2),
        zorder=11,
    )
    ax.text(
        0.012, 0.94, panel_label, transform=ax.transAxes,
        ha="left", va="top", fontsize=14, fontweight="bold",
        color="#272C2F", zorder=12,
    )
    ax.text(
        0.985, 0.94, SCENARIO_LABELS[scenario], transform=ax.transAxes,
        ha="right", va="top", fontsize=12, fontweight="bold",
        color=color, zorder=12,
    )
    ax.set_xlim(pd.Timestamp("2026-01-01"), pd.Timestamp("2100-12-31"))
    ax.set_ylim(*ylim)
    style_axis(ax)


def make_figure(projection: pd.DataFrame, stats: pd.DataFrame) -> tuple[plt.Figure, tuple[float, float]]:
    lower, upper, step = shared_limits(projection, stats)
    fig, axes = plt.subplots(
        4, 1, figsize=(9.2, 10.8), sharex=True, sharey=True,
        gridspec_kw={"hspace": 0.13},
    )
    for ax, scenario, label in zip(axes, SCENARIOS, "abcd"):
        draw_panel(ax, projection, stats, scenario, label, (lower, upper))
        ax.set_yticks(
            np.arange(math.ceil(lower / step) * step, upper + step * 0.5, step)
        )
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(10))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Year", labelpad=5)
    fig.supylabel("TWSA scenario projection (mm)", x=0.021, fontsize=14)

    handles = [
        Line2D([0], [0], color="#6E8292", lw=0.7, alpha=0.35,
               label="Central monthly trajectory"),
        Line2D([0], [0], color="#4F6573", lw=2.6,
               label="Central MA12 (assumption A)"),
        Line2D([0], [0], color="#363636", lw=1.15, ls=(0, (3.5, 2.4)),
               label="Climate-sensitive MA12"),
        Line2D([0], [0], color="#4F6573", lw=0.90, alpha=0.60,
               label="MA12 P25–P75 boundaries"),
        Line2D([0], [0], color="#AEB4B8", lw=0.75, ls=(0, (3.0, 2.5)),
               label="MA12 full-envelope boundaries"),
    ]
    handles = [handles[index] for index in (0, 3, 1, 4, 2)]
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.008),
        ncol=3, frameon=False, handlelength=2.7, handletextpad=0.65,
        columnspacing=1.55, borderaxespad=0,
    )
    fig.subplots_adjust(left=0.105, right=0.985, top=0.985, bottom=0.105)
    return fig, (lower, upper)


def write_verification(
    source: pd.DataFrame,
    projection: pd.DataFrame,
    stats: pd.DataFrame,
    ylim: tuple[float, float],
) -> None:
    rows = []
    for scenario in SCENARIOS:
        all_paths = projection[projection["scenario"].eq(scenario)]
        central = all_paths[all_paths["assumption"].eq(CENTRAL)].sort_values("date")
        bounds = stats[stats["scenario"].eq(scenario)].sort_values("date")
        endpoint = float(central.dropna(subset=["total_twsa_ma12"]).iloc[-1]["total_twsa_ma12"])
        original_central = source[
            source["scenario"].eq(scenario) & source["assumption"].eq(CENTRAL)
        ].sort_values("date")
        rows.append(
            {
                "scenario": scenario,
                "formal_label": SCENARIO_LABELS[scenario],
                "assumption_count_used_for_ranges": all_paths["assumption"].nunique(),
                "central_assumption": CENTRAL,
                "central_monthly_row_count": len(central),
                "central_monthly_minimum_mm": central["total_twsa_reconstruction_mm"].min(),
                "central_monthly_maximum_mm": central["total_twsa_reconstruction_mm"].max(),
                "central_ma12_valid_count": central["total_twsa_ma12"].notna().sum(),
                "central_ma12_2100_value_mm": endpoint,
                "central_endpoint_matches_reference": round(endpoint, 2) == REFERENCE_ENDPOINTS[scenario],
                "ma12_p25_minimum_mm": bounds["p25_ma12"].min(),
                "ma12_p75_maximum_mm": bounds["p75_ma12"].max(),
                "ma12_full_envelope_minimum_mm": bounds["envelope_min_ma12"].min(),
                "ma12_full_envelope_maximum_mm": bounds["envelope_max_ma12"].max(),
                "central_monthly_plotted": True,
                "central_ma12_plotted": True,
                "climate_ma12_plotted": True,
                "ma12_p25_p75_boundaries_plotted": True,
                "ma12_full_envelope_boundaries_plotted": True,
                "individual_B_to_E_paths_plotted": False,
                "source_values_unchanged": np.array_equal(
                    central["total_twsa_reconstruction_mm"].to_numpy(),
                    original_central["total_twsa_reconstruction_mm"].to_numpy(),
                ),
                "shared_y_min_mm": ylim[0],
                "shared_y_max_mm": ylim[1],
                "source_file": str(SOURCE_CSV),
                "source_sha256": sha256(SOURCE_CSV),
            }
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(OUT / "Fig6_range_check.csv", index=False, encoding="utf-8-sig")
    if len(audit) != 4 or not audit["central_endpoint_matches_reference"].all():
        raise AssertionError("Central endpoint QA failed")
    if not audit["source_values_unchanged"].all():
        raise AssertionError("Central monthly source values changed")
    if audit["individual_B_to_E_paths_plotted"].any():
        raise AssertionError("B-E individual paths must not be plotted in the main figure")
    if not audit["assumption_count_used_for_ranges"].eq(5).all():
        raise AssertionError("Range statistics do not use all five assumptions")


def write_caption_and_note(ylim: tuple[float, float]) -> None:
    caption = (
        "Fig. 6. Conditional TWSA scenario projections under explicit residual assumptions. "
        "Thin SSP-coloured lines show the monthly selected central trajectories under assumption A, "
        "and the thicker lines show their trailing 12-month moving averages. Dark-gray dashed lines "
        "denote the climate-sensitive MA12 components. The inner SSP-coloured boundary lines indicate "
        "the P25–P75 range calculated across the residual-assumption-conditioned MA12 trajectories, whereas the "
        "outer gray dashed lines denote their full MA12 envelope. Values at the line endpoints are the selected "
        "central MA12 projections for 2100. Detailed monthly and MA12 trajectories under individual residual "
        "assumptions A–E are provided in Supplementary Fig. S7. The displayed ranges are assumption-based and do not "
        "represent confidence intervals, prediction intervals, or deterministic forecasts."
    )
    (OUT / "figure_caption_en.txt").write_text(caption, encoding="utf-8")
    note = f"""# Fig. 6 source-data note

- The main-text figure directly plots only assumption A as monthly and trailing-MA12 trajectories.
- Assumptions B–E contribute only to MA12 P25–P75 and MA12 full-envelope boundary calculations; their individual trajectories remain in Supplementary Fig. S7.
- The source projection file was read without modifying any values.
- No model was retrained and no future projection was rerun.
- The shared y-axis range is {ylim[0]:.0f} to {ylim[1]:.0f} mm and covers all five assumptions used for the displayed boundaries.
- No range fill is drawn. All range information is represented by low-frequency MA12 boundary lines.
- The displayed boundaries are assumption-based, not probabilistic confidence or prediction intervals.
"""
    (OUT / "visual_revision_note.md").write_text(note, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    setup_style()
    source, projection, stats = load_data()
    fig, ylim = make_figure(projection, stats)
    fig.savefig(STEM.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(STEM.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(STEM.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(
        STEM.with_suffix(".tif"), dpi=600, bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
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
    write_verification(source, projection, stats, ylim)
    write_caption_and_note(ylim)
    print(f"Source: {SOURCE_CSV}")
    print(f"Source SHA-256: {sha256(SOURCE_CSV)}")
    print("Directly plotted paths: central monthly=4, central MA12=4, climate MA12=4")
    print("Individual B-E paths plotted: 0")
    print("Displayed range boundaries use MA12 paths from all five assumptions for every SSP")
    print(f"Shared y-axis: {ylim[0]:.0f} to {ylim[1]:.0f} mm")
    print(f"Outputs: {OUT}")


if __name__ == "__main__":
    main()
