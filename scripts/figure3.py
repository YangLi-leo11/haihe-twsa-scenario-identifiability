# -*- coding: utf-8 -*-
"""Reproduce the reported robustness figure from the included source tables.

Only the included processed summary and feature-sensitivity tables are read.
No model or experiment is run here.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.text
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import cee_figure_style as cfs


REPO_ROOT = Path(__file__).resolve().parents[1]
V3 = REPO_ROOT / "data" / "processed" / "fig3"
V25 = V3
OUT = REPO_ROOT / "figures" / "reproduced"
OUT.mkdir(parents=True, exist_ok=True)

MAIN_HORIZONS = list(range(12, 85, 6))
ANCHOR_HORIZONS = [12, 24, 36, 48, 60, 72, 84]
OLD_HORIZONS = [12, 24, 36, 60, 84]
DRIVERS = ["P", "T", "ET", "Q"]
FEATURE_LABELS = {"P": "Precipitation (P)", "T": "Temperature (T)", "ET": "Evapotranspiration\n(ET)", "Q": "Runoff (Q)"}
COLORS = {
    "r": cfs.MODEL, "r_fill": "#DCE8EE", "r_raw": "#8CA9BA",
    "skill": "#3C8585", "skill_fill": "#DCEAE8", "skill_raw": "#8BB2AC",
    "ink": cfs.TEXT, "muted": "#727B80",
    "frame": cfs.FRAME, "grid": cfs.GRID, "zero": cfs.ZERO_LINE,
}


def read_sources() -> dict[str, pd.DataFrame]:
    paths = {
        "a_case": V25 / "horizon_6month_case_metrics.csv",
        "a_summary": V25 / "horizon_6month_summary.csv",
        "b_case": V25 / "persistence_6month_case_metrics.csv",
        "b_summary": V25 / "persistence_6month_summary.csv",
        "sensitivity_summary": V3 / "experiment_B_driver_ablation_summary_v3.csv",
        "sensitivity_sign": V3 / "experiment_B_driver_ablation_sign_consistency_v3.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required CSV files: " + "; ".join(missing))
    data = {key: pd.read_csv(path) for key, path in paths.items()}
    for key in ("a_case", "b_case", "a_summary", "b_summary"):
        data[key]["forecast_horizon_months"] = data[key]["forecast_horizon_months"].astype(int)
    return data


def validate_sources(data: dict[str, pd.DataFrame]) -> dict[str, object]:
    a_case, b_case = data["a_case"], data["b_case"]
    a_summary, b_summary = data["a_summary"], data["b_summary"]
    a_main = a_case[a_case["forecast_horizon_months"].isin(MAIN_HORIZONS)]
    b_main = b_case[b_case["forecast_horizon_months"].isin(MAIN_HORIZONS)]
    expected = {h: 5 for h in MAIN_HORIZONS}
    a_counts = a_main.groupby("forecast_horizon_months").size().to_dict()
    b_counts = b_main.groupby("forecast_horizon_months").size().to_dict()
    origins = sorted(a_main["forecast_origin"].astype(str).unique().tolist())
    old_a = pd.read_csv(V3 / "experiment_A_matched_origin_summary_v3.csv")
    old_b = pd.read_csv(V3 / "experiment_C_persistence_summary_v3.csv")
    old_a["forecast_horizon_months"] = old_a["forecast_horizon_months"].astype(int)
    old_b["forecast_horizon_months"] = old_b["forecast_horizon_months"].astype(int)
    source_a = a_summary[a_summary["forecast_horizon_months"].isin(OLD_HORIZONS)].set_index("forecast_horizon_months").loc[OLD_HORIZONS]
    source_b = b_summary[b_summary["forecast_horizon_months"].isin(OLD_HORIZONS)].set_index("forecast_horizon_months").loc[OLD_HORIZONS]
    frozen_a = old_a.set_index("forecast_horizon_months").loc[OLD_HORIZONS]
    frozen_b = old_b.set_index("forecast_horizon_months").loc[OLD_HORIZONS]
    reproduced_a = np.allclose(source_a[["median_R", "P25_R", "P75_R"]].to_numpy(float), frozen_a[["median_R", "P25_R", "P75_R"]].to_numpy(float), rtol=1e-10, atol=1e-12)
    reproduced_b = np.allclose(source_b[["median_SS_persistence", "P25_SS_persistence", "P75_SS_persistence"]].to_numpy(float), frozen_b[["median_SS_persistence", "P25_SS_persistence", "P75_SS_persistence"]].to_numpy(float), rtol=1e-10, atol=1e-12)
    leakage = int(a_case.get("future_leakage_count", pd.Series(dtype=float)).fillna(0).sum()) + int(b_case.get("future_leakage_count", pd.Series(dtype=float)).fillna(0).sum())
    self_match = int(a_case.get("target_self_match_count", pd.Series(dtype=float)).fillna(0).sum()) + int(b_case.get("target_self_match_count", pd.Series(dtype=float)).fillna(0).sum())
    six_a = a_summary.loc[a_summary["forecast_horizon_months"] == 6].iloc[0]
    six_b = b_summary.loc[b_summary["forecast_horizon_months"] == 6].iloc[0]
    return {
        "a_case": a_main.copy(), "b_case": b_main.copy(), "origins": origins,
        "a_counts": a_counts, "b_counts": b_counts,
        "all_n_equal_5": a_counts == expected and b_counts == expected and len(origins) == 5,
        "all_summary_rows_present": set(a_summary["forecast_horizon_months"]) >= set([6] + MAIN_HORIZONS) and set(b_summary["forecast_horizon_months"]) >= set([6] + MAIN_HORIZONS),
        "existing_values_reproduced": bool(reproduced_a and reproduced_b),
        "future_leakage_count": leakage, "self_match_count": self_match,
        "six_month_R": float(six_a["median_R"]), "six_month_skill": float(six_b["median_SS_persistence"]),
    }


def style_axis(ax: plt.Axes, *, grid_axis: str | None = None) -> None:
    cfs.style_cartesian(ax, grid_axis=grid_axis)


def draw_panel_a(ax: plt.Axes, data: dict[str, pd.DataFrame]) -> None:
    summary = data["a_summary"].set_index("forecast_horizon_months").loc[MAIN_HORIZONS]
    case = data["a_case"]
    x = np.asarray(MAIN_HORIZONS, dtype=float)
    style_axis(ax, grid_axis="y")
    ax.axhline(0.5, color=COLORS["zero"], linewidth=0.8, linestyle=(0, (3.0, 2.2)), zorder=1)
    ax.fill_between(x, summary["P25_R"].to_numpy(float), summary["P75_R"].to_numpy(float), color=COLORS["r_fill"], alpha=0.72, zorder=1)
    for h in MAIN_HORIZONS:
        values = case.loc[case["forecast_horizon_months"] == h, "R"].to_numpy(float)
        jitter = np.linspace(-0.62, 0.62, len(values))
        ax.scatter(h + jitter, values, s=13, color=COLORS["r_raw"], alpha=0.28, linewidths=0, zorder=2)
    median = summary["median_R"].to_numpy(float)
    ax.plot(x, median, color=COLORS["r"], linewidth=2.1, zorder=3)
    ax.scatter(x, median, s=23, color=COLORS["r"], edgecolor="white", linewidth=0.65, zorder=4)
    ax.set_xlim(10, 86); ax.set_ylim(0.45, 1.00)
    major, minor = [12, 24, 36, 48, 60, 72, 84], [18, 30, 42, 54, 66, 78]
    ax.set_xticks(major); ax.set_xticks(minor, minor=True); ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.tick_params(axis="x", which="minor", length=1.4, color=COLORS["muted"])
    ax.set_xlabel("Forecast horizon (months)"); ax.set_ylabel("Correlation, R")
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=COLORS["r_raw"], markeredgecolor="none", alpha=0.82, markersize=4.2),
        Line2D([0], [0], color=COLORS["r"], linewidth=2.1, marker="o", markerfacecolor=COLORS["r"], markeredgecolor="white", markeredgewidth=0.6, markersize=4.2),
        Patch(facecolor=COLORS["r_fill"], edgecolor="none", alpha=0.85),
    ]
    ax.legend(handles, ["Matched-origin cases (n=5)", "Median", "P25–P75"], ncol=3, loc="upper right", bbox_to_anchor=(0.992, 0.986), **cfs.legend_kwargs(handlelength=1.15, handletextpad=0.35, columnspacing=0.8, borderaxespad=0))


def draw_panel_b(ax: plt.Axes, data: dict[str, pd.DataFrame]) -> None:
    summary = data["b_summary"].set_index("forecast_horizon_months").loc[MAIN_HORIZONS]
    case = data["b_case"]
    x = np.asarray(MAIN_HORIZONS, dtype=float)
    style_axis(ax, grid_axis="y")
    ax.axhline(0.0, color=COLORS["zero"], linewidth=0.8, zorder=1)
    for h in MAIN_HORIZONS:
        values = case.loc[case["forecast_horizon_months"] == h, "SS_persistence"].to_numpy(float)
        jitter = np.linspace(-0.62, 0.62, len(values))
        ax.scatter(h + jitter, values, s=12, color=COLORS["skill_raw"], alpha=0.28, linewidths=0, zorder=2)
        row = summary.loc[h]
        median = float(row["median_SS_persistence"])
        p25 = float(row["P25_SS_persistence"])
        p75 = float(row["P75_SS_persistence"])
        ax.vlines(h, 0.0, median, color=COLORS["skill_fill"], linewidth=1.0, zorder=1)
        ax.errorbar(h, median, yerr=[[median - p25], [p75 - median]], fmt="none", ecolor=COLORS["skill"], elinewidth=1.05, capsize=2.0, capthick=0.8, zorder=3)
        ax.scatter(h, median, s=25, color=COLORS["skill"], edgecolor="white", linewidth=0.7, zorder=4)
    ax.set_xlim(10, 86); ax.set_ylim(-0.08, 0.74)
    major, minor = [12, 24, 36, 48, 60, 72, 84], [18, 30, 42, 54, 66, 78]
    ax.set_xticks(major); ax.set_xticks(minor, minor=True); ax.set_yticks([0.0, 0.2, 0.4, 0.6])
    ax.tick_params(axis="x", which="minor", length=1.4, color=COLORS["muted"])
    ax.set_xlabel("Forecast horizon (months)"); ax.set_ylabel("Skill relative to persistence")


def draw_panel_c_stacked(ax_r: plt.Axes, ax_rmse: plt.Axes, data: dict[str, pd.DataFrame]) -> None:
    summary = data["sensitivity_summary"]
    sign = data["sensitivity_sign"]
    fractions = []
    for driver in DRIVERS:
        c60 = int(sign.loc[(sign["driver_group"] == driver) & (sign["forecast_horizon_months"] == 60), "joint_degradation_count"].iloc[0])
        c84 = int(sign.loc[(sign["driver_group"] == driver) & (sign["forecast_horizon_months"] == 84), "joint_degradation_count"].iloc[0])
        fractions.append(f"{c60}/5 | {c84}/5")

    x = np.arange(len(DRIVERS), dtype=float)
    bar_width = 0.27
    offsets = {60: -0.155, 84: 0.155}
    for ax in (ax_r, ax_rmse):
        style_axis(ax, grid_axis="y")
        ax.axhline(0.0, color=COLORS["zero"], linewidth=0.8, zorder=1)
        ax.set_xlim(-0.55, 3.55)
        ax.set_xticks(x)

    r_values = summary[["median_delta_R", "P25_delta_R", "P75_delta_R"]].to_numpy(float).ravel()
    e_values = summary[["median_delta_RMSE_mm", "P25_delta_RMSE_mm", "P75_delta_RMSE_mm"]].to_numpy(float).ravel()
    r_limit = max(0.42, max(abs(float(np.nanmin(r_values))), abs(float(np.nanmax(r_values)))) * 1.24)
    e_limit = max(12.8, max(abs(float(np.nanmin(e_values))), abs(float(np.nanmax(e_values)))) * 1.22)
    ax_r.set_ylim(-r_limit, r_limit)
    ax_rmse.set_ylim(-e_limit, e_limit)
    ax_r.set_yticks([-0.4, -0.2, 0.0, 0.2, 0.4])
    ax_rmse.set_yticks([-10.0, 0.0, 10.0])

    for i, driver in enumerate(DRIVERS):
        for horizon in (60, 84):
            color = cfs.PREDICTOR[driver]
            alpha = 0.90 if horizon == 60 else 0.52
            xpos = x[i] + offsets[horizon]
            row = summary.loc[(summary["driver_group"] == driver) & (summary["forecast_horizon_months"] == horizon)].iloc[0]
            vr, p25r, p75r = float(row["median_delta_R"]), float(row["P25_delta_R"]), float(row["P75_delta_R"])
            ve, p25e, p75e = float(row["median_delta_RMSE_mm"]), float(row["P25_delta_RMSE_mm"]), float(row["P75_delta_RMSE_mm"])
            ecolor = matplotlib.colors.to_rgba(color, max(alpha, 0.62))
            ax_r.bar(xpos, vr, width=bar_width, color=color, alpha=alpha, edgecolor=color, linewidth=0.30, zorder=2)
            ax_rmse.bar(xpos, ve, width=bar_width, color=color, alpha=alpha, edgecolor=color, linewidth=0.30, zorder=2)
            ax_r.errorbar(xpos, vr, yerr=[[vr - p25r], [p75r - vr]], fmt="none", ecolor=ecolor, elinewidth=0.88, capsize=1.9, capthick=0.78, zorder=4)
            ax_rmse.errorbar(xpos, ve, yerr=[[ve - p25e], [p75e - ve]], fmt="none", ecolor=ecolor, elinewidth=0.88, capsize=1.9, capthick=0.78, zorder=4)

    ax_r.tick_params(axis="x", labelbottom=False, bottom=False)
    compact_feature = ["P", "T", "ET", "Q"]
    ax_rmse.set_xticklabels([f"{feature}\n{fraction}" for feature, fraction in zip(compact_feature, fractions)], fontsize=7.4, linespacing=0.92)
    ax_r.set_ylabel("ΔR")
    ax_rmse.set_ylabel("ΔRMSE (mm)")
    ax_rmse.set_xlabel("Feature group; consistent origins (60 m | 84 m)", labelpad=2.0)


def draw_panel_d_combined(ax: plt.Axes, data: dict[str, pd.DataFrame]) -> None:
    origins = ["2014-01-01", "2015-01-01", "2016-01-01", "2017-01-01", "2018-01-01"]
    horizons = ANCHOR_HORIZONS
    r = matrix_from_case(data["a_case"], "R", origins, horizons)
    s = matrix_from_case(data["b_case"], "SS_persistence", origins, horizons)
    r_cmap = LinearSegmentedColormap.from_list("r_scale_v33", ["#F6F4EF", "#E3E9EE", "#BFCFDE", "#89A7C7", "#4F79B8"])
    s_cmap = LinearSegmentedColormap.from_list("skill_scale_v33", ["#C95A4D", "#E5B4A9", "#F4F1ED", "#B7D5C7", "#3E8A70"])
    r_norm = Normalize(0.45, 1.0)
    s_norm = TwoSlopeNorm(vmin=-0.10, vcenter=0.0, vmax=0.72)
    y_edges = np.arange(6, dtype=float) - 0.5
    x_r_edges = np.arange(8, dtype=float) - 0.5
    x_s_edges = np.arange(8, dtype=float) + 7.5
    ax.pcolormesh(x_r_edges, y_edges, r, cmap=r_cmap, norm=r_norm, shading="flat", edgecolors="white", linewidth=0.62, zorder=1)
    ax.pcolormesh(x_s_edges, y_edges, s, cmap=s_cmap, norm=s_norm, shading="flat", edgecolors="white", linewidth=0.62, zorder=1)
    for matrix, x_offset, cmap, norm in ((r, 0, r_cmap, r_norm), (s, 8, s_cmap, s_norm)):
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                rgba = cmap(norm(value))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                ax.text(x_offset + j, i, f"{value:.2f}", ha="center", va="center", fontsize=5.75, color="white" if luminance < 0.58 else COLORS["ink"], zorder=3)
    ax.set_xlim(-0.5, 14.5)
    ax.set_ylim(4.5, -0.5)
    ax.set_xticks(list(range(7)) + list(range(8, 15)))
    ax.set_xticklabels([str(h) for h in horizons] * 2, fontsize=6.25)
    ax.set_yticks(np.arange(5))
    ax.set_yticklabels([pd.Timestamp(o).year for o in origins], fontsize=7.15)
    ax.tick_params(axis="x", length=2.0, width=0.5, pad=1.2, colors=COLORS["ink"])
    ax.tick_params(axis="y", length=0, pad=1.8, colors=COLORS["ink"])
    ax.axvspan(6.5, 7.5, color="white", zorder=2)
    ax.axvline(7.0, color="#D6DDE0", linewidth=0.62, zorder=3)
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_color(COLORS["frame"]); spine.set_linewidth(0.74)
    ax.set_facecolor("white")
    ax.set_ylabel("Origin year", fontsize=8.3, labelpad=2.0)
    ax.text(3.0 / 15.0, -0.150, "R · forecast horizon (months)", transform=ax.transAxes, ha="center", va="top", fontsize=7.05, color=COLORS["ink"])
    ax.text(11.0 / 15.0, -0.150, "Skill · forecast horizon (months)", transform=ax.transAxes, ha="center", va="top", fontsize=7.05, color=COLORS["ink"])


def draw_panel_d_split(ax_r: plt.Axes, ax_skill: plt.Axes, data: dict[str, pd.DataFrame]) -> None:
    origins = ["2014-01-01", "2015-01-01", "2016-01-01", "2017-01-01", "2018-01-01"]
    horizons = ANCHOR_HORIZONS
    r = matrix_from_case(data["a_case"], "R", origins, horizons)
    s = matrix_from_case(data["b_case"], "SS_persistence", origins, horizons)
    r_cmap = LinearSegmentedColormap.from_list("r_scale_harmonized", ["#F5F3EE", "#DDE7EB", "#A7C4D0", "#6E9CB1", cfs.MODEL])
    s_cmap = LinearSegmentedColormap.from_list("skill_scale_harmonized", ["#F5F3EE", "#DCE9E6", "#A9CEC6", "#72AFA4", "#3C8585"])
    annotated_heatmap(ax_r, r, horizons, r_cmap, Normalize(0.45, 1.0), origins)
    annotated_heatmap(ax_skill, s, horizons, s_cmap, TwoSlopeNorm(vmin=-0.10, vcenter=0.0, vmax=0.72), origins)
    ax_skill.tick_params(axis="y", labelleft=False, left=False)
    ax_r.set_ylabel("Origin year")
    ax_r.set_xlabel("R · forecast horizon (months)", labelpad=2.8)
    ax_skill.set_xlabel("Skill · forecast horizon (months)", labelpad=2.8)


def matrix_from_case(case: pd.DataFrame, value_col: str, origins: list[str], horizons: list[int]) -> np.ndarray:
    table = case.pivot(index="forecast_origin", columns="forecast_horizon_months", values=value_col)
    return table.loc[origins, horizons].to_numpy(float)


def annotated_heatmap(ax: plt.Axes, matrix: np.ndarray, horizons: list[int], cmap, norm, origins: list[str]) -> plt.cm.ScalarMappable:
    image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest", origin="upper")
    ax.set_xticks(np.arange(len(horizons))); ax.set_xticklabels([str(h) for h in horizons], fontsize=7.1, color=COLORS["ink"])
    ax.tick_params(axis="x", labeltop=False, labelbottom=True, top=False, bottom=True, pad=1.0, length=2.0)
    ax.set_yticks(np.arange(len(origins))); ax.set_yticklabels([pd.Timestamp(o).year for o in origins], fontsize=7.1, color=COLORS["ink"])
    ax.tick_params(axis="y", length=0, pad=1.3)
    ax.set_xticks(np.arange(-0.5, len(horizons), 1), minor=True); ax.set_yticks(np.arange(-0.5, len(origins), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.60); ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]; rgba = cmap(norm(value)); luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7.0, color="white" if luminance < 0.58 else COLORS["ink"], zorder=3)
    for spine in ax.spines.values(): spine.set_color(COLORS["frame"]); spine.set_linewidth(0.8)
    return image


def draw_panel_d(ax_r: plt.Axes, ax_skill: plt.Axes, cax_r: plt.Axes, cax_s: plt.Axes, data: dict[str, pd.DataFrame], fig: plt.Figure) -> None:
    origins = ["2014-01-01", "2015-01-01", "2016-01-01", "2017-01-01", "2018-01-01"]
    horizons = ANCHOR_HORIZONS
    r = matrix_from_case(data["a_case"], "R", origins, horizons); s = matrix_from_case(data["b_case"], "SS_persistence", origins, horizons)
    r_cmap = LinearSegmentedColormap.from_list("r_scale_v31", ["#F6F4EF", "#E3E9EE", "#BFCFDE", "#89A7C7", "#4F79B8"])
    s_cmap = LinearSegmentedColormap.from_list("skill_scale_v31", ["#C95A4D", "#E5B4A9", "#F4F1ED", "#B7D5C7", "#3E8A70"])
    image_r = annotated_heatmap(ax_r, r, horizons, r_cmap, Normalize(0.45, 1.0), origins)
    image_s = annotated_heatmap(ax_skill, s, horizons, s_cmap, TwoSlopeNorm(vmin=-0.10, vcenter=0.0, vmax=0.72), origins)
    ax_skill.set_ylabel(""); ax_skill.tick_params(axis="y", labelleft=False, left=False)
    for image, cax, ticks, label in (
        (image_r, cax_r, [0.5, 0.7, 0.9, 1.0], "Correlation, R"),
        (image_s, cax_s, [-0.1, 0.0, 0.2, 0.4, 0.6], "Skill relative to persistence"),
    ):
        cbar = fig.colorbar(image, cax=cax, orientation="horizontal")
        cbar.set_ticks(ticks); cbar.ax.tick_params(labelsize=6.2, length=1.4, pad=1.0)
        cbar.set_label(label, fontsize=7.1, color=COLORS["ink"], labelpad=1.8)
        cbar.ax.xaxis.set_label_position("top")
        cbar.outline.set_linewidth(0.55); cbar.outline.set_edgecolor(COLORS["frame"])


def audit_text(fig: plt.Figure) -> tuple[int, int]:
    fig.canvas.draw(); renderer = fig.canvas.get_renderer(); width, height = renderer.width, renderer.height
    texts = [t for t in fig.findobj(match=matplotlib.text.Text) if t.get_visible() and t.get_text()]
    boxes = []; outside = 0
    for t in texts:
        box = t.get_window_extent(renderer=renderer)
        if not np.all(np.isfinite([box.x0, box.y0, box.x1, box.y1])): continue
        if box.x0 < 0 or box.y0 < 0 or box.x1 > width or box.y1 > height: outside += 1
        boxes.append(box)
    overlap = 0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            first, second = boxes[i], boxes[j]
            ix = max(0.0, min(first.x1, second.x1) - max(first.x0, second.x0)); iy = max(0.0, min(first.y1, second.y1) - max(first.y0, second.y0))
            if ix * iy > 6.0: overlap += 1
    return outside, overlap


def make_figure(data: dict[str, pd.DataFrame], checks: dict[str, object]) -> tuple[int, int]:
    cfs.apply_style()
    fig = plt.figure(figsize=(cfs.FINAL_WIDTH_MM * cfs.MM, 4.6678), facecolor="white")

    # Every visible border below belongs to a real data axis. No framing axes or
    # decorative rectangles are used to simulate panel alignment.
    # Top and bottom rows use independent horizontal geometry. Alignment is
    # enforced within each row, not by forcing one column split through both.
    a_box = (0.075, 0.565, 0.480, 0.345)
    b_box = (0.615, 0.565, 0.350, 0.345)
    c_top_box = (0.075, 0.321, 0.340, 0.153)
    c_bottom_box = (0.075, 0.140, 0.340, 0.153)
    # Shift the matrix group right and trim each matrix slightly so the d1
    # Origin year label clears panel c without changing either matrix value.
    d1_box = (0.490, 0.140, 0.215, 0.334)
    d2_box = (0.750, 0.140, 0.215, 0.334)
    ax_a = fig.add_axes(a_box); draw_panel_a(ax_a, data)
    ax_b = fig.add_axes(b_box); draw_panel_b(ax_b, data)

    ax_cr = fig.add_axes(c_top_box)
    ax_ce = fig.add_axes(c_bottom_box, sharex=ax_cr)
    draw_panel_c_stacked(ax_cr, ax_ce, data)
    c_handles = [
        Patch(facecolor=cfs.TEXT, edgecolor=cfs.TEXT, alpha=0.90, label="60 m"),
        Patch(facecolor=cfs.TEXT, edgecolor=cfs.TEXT, alpha=0.52, label="84 m"),
    ]
    ax_cr.legend(c_handles, ["60 m", "84 m"], ncol=2, loc="upper right", bbox_to_anchor=(0.992, 0.975), handleheight=0.65, **cfs.legend_kwargs(handlelength=0.9, handletextpad=0.32, columnspacing=0.85, borderaxespad=0))

    ax_dr = fig.add_axes(d1_box)
    ax_ds = fig.add_axes(d2_box, sharey=ax_dr)
    draw_panel_d_split(ax_dr, ax_ds, data)

    for letter, axis in (("a", ax_a), ("b", ax_b), ("c", ax_cr)):
        cfs.panel_label(axis, letter)
    cfs.panel_label(ax_dr, "d", x=-0.085, y=1.02)
    outside, overlap = audit_text(fig)
    stem = OUT / "Fig3_CEE_harmonized_v2"
    cfs.save_figure(fig, stem)
    plt.close(fig)
    return outside, overlap


def write_verification(checks: dict[str, object], outside: int, overlap: int) -> None:
    text = f"""# Figure 3 rendering and numerical verification

- Scientific source values changed: no
- Model or experiment rerun: no
- Forecast horizons: {','.join(map(str, MAIN_HORIZONS))} months
- Common origins: {len(checks['origins'])}
- Future-leakage count: {checks['future_leakage_count']}
- Self-match count: {checks['self_match_count']}
- Included summary values reproduced: {'yes' if checks['existing_values_reproduced'] else 'no'}
- Text outside canvas: {outside}
- Text-overlap pairs: {overlap}
- Render verification: {'PASS' if outside == 0 and overlap == 0 else 'FAIL'}
"""
    (OUT / "Fig3_render_verification.md").write_text(text, encoding="utf-8")

def write_caption() -> None:
    caption = (
        "Panel a summarizes forecast correlation from 12 to 84 months at 6-month intervals using a median-point profile, interquartile ribbon and five visible common matched-origin cases. "
        "Panel b independently displays persistence-relative skill at each horizon using stems, interquartile whiskers, median points and the same matched-origin cases. "
        "Panel c reports paired 60- and 84-month feature-sensitivity responses in two aligned grouped-bar axes; compact category labels also report consistent-origin counts (60 m | 84 m). "
        "Panel d places origin-level correlation and persistence-relative skill in two blocks within one aligned matrix axis at seven representative annual-scale horizons (12, 24, 36, 48, 60, 72 and 84 months); correlation uses a sequential blue scale, whereas signed skill is centered at zero with a muted red-white-green scale. "
        "The 6-month diagnostic is retained in the supplementary robustness verification but excluded from the main figure because it represents a distinct very-short-window regime."
    )
    (OUT / "Fig3_caption.txt").write_text(caption + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = read_sources(); checks = validate_sources(data)
    if not checks["all_n_equal_5"]: raise ValueError("The horizon subset does not contain five cases per horizon")
    if not checks["all_summary_rows_present"]: raise ValueError("The summary grid is incomplete")
    if not checks["existing_values_reproduced"]: raise ValueError("Included 12/24/36/60/84 results were not reproduced")
    outside, overlap = make_figure(data, checks)
    write_verification(checks, outside, overlap)
    write_caption()
    print("FIGURE3_RENDERING_COMPLETE")
    print(f"output_dir={OUT}")
    print(f"text_outside={outside}; text_overlap={overlap}")


if __name__ == "__main__":
    main()
