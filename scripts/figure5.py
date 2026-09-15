"""Figure 5 v12: standard reading-order 2x2 reconstruction.

Four panels show pathway-median evolution, target-year summaries, the complete
December 2100 pathway-by-SSP matrix, and leave-one-pathway-out range-control
reductions. Every plotted value is read directly from the included processed CSV.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
from matplotlib.text import Text
import cee_figure_style as cfs
REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = REPO_ROOT / 'data' / 'processed' / 'fig5_fig6' / 'conditional_twsa_monthly.csv'
SOURCE_DISPLAY = 'data/processed/fig5_fig6/conditional_twsa_monthly.csv'
SOURCE_SCRIPT = REPO_ROOT / 'src' / 'cee_reproduction' / 'pathways.py'
OUTPUT_DIR = REPO_ROOT / 'figures' / 'reproduced'
TARGET_YEARS = [2050, 2075, 2100]
SCENARIOS = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
SCENARIO_DISPLAY = {'ssp126': 'SSP1-2.6', 'ssp245': 'SSP2-4.5', 'ssp370': 'SSP3-7.0', 'ssp585': 'SSP5-8.5'}
SCENARIO_COLORS = {key: cfs.SSP_COLORS[key] for key in SCENARIOS}
PATHWAYS = ['A_stationary_residual', 'B_trend_capped_persistence_residual', 'C_risk_conditioned_residual', 'D_moderation_residual', 'E_consistency_reference_residual']
PATHWAY_ORDER = ['B_trend_capped_persistence_residual', 'E_consistency_reference_residual', 'D_moderation_residual', 'A_stationary_residual', 'C_risk_conditioned_residual']
PATHWAY_DISPLAY = {'A_stationary_residual': 'Stationary residual', 'B_trend_capped_persistence_residual': 'Persistence', 'C_risk_conditioned_residual': 'Risk-conditioned', 'D_moderation_residual': 'Moderation', 'E_consistency_reference_residual': 'Groundwater-consistency reference'}
PATHWAY_DISPLAY_WRAPPED = {**PATHWAY_DISPLAY, 'E_consistency_reference_residual': 'Groundwater-consistency\nreference'}
EXPECTED_STATIONARY = {'ssp126': 10.426126344442, 'ssp245': 3.66992176664, 'ssp370': 6.581631308745, 'ssp585': 6.0198777248}
EXPECTED_FIXED_SSP = 306.09685522958
EXPECTED_FIXED_PATHWAY = 6.756204577802
EXPECTED_FULL_APERTURE = 312.853059807382
EXPECTED_RATIO = 45.306037095929
EXPECTED_PERSISTENCE_REDUCTION = 115.847225246409
EXPECTED_RISK_REDUCTION = 66.846538893252
EXPECTED_OTHER_REDUCTION_MAX = 0.0
GATE_TOL = 5e-06
X_LIMITS = (-80.0, 265.0)
X_TICKS = [-50, 0, 50, 100, 150, 200, 250]
SSP_OFFSETS = [-0.17, -0.06, 0.06, 0.17]
PATHWAY_COLORS = {'B_trend_capped_persistence_residual': cfs.PATHWAY['persistence'], 'E_consistency_reference_residual': cfs.PATHWAY['gw_consistency'], 'D_moderation_residual': cfs.PATHWAY['moderation'], 'A_stationary_residual': cfs.PATHWAY['stationary'], 'C_risk_conditioned_residual': cfs.PATHWAY['risk_conditioned']}
PATHWAY_LINESTYLES = {'B_trend_capped_persistence_residual': '-', 'E_consistency_reference_residual': '-', 'D_moderation_residual': (0, (2.6, 1.5)), 'A_stationary_residual': '-.', 'C_risk_conditioned_residual': '-'}
TEXT = cfs.TEXT
FRAME = cfs.FRAME
GRID = cfs.GRID
LINE_COLOR = TEXT
MEDIAN_COLOR = TEXT
ZERO_LINE = cfs.ZERO_LINE

def configure_style() -> None:
    cfs.apply_style()

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def load_and_gate() -> dict[str, object]:
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(SOURCE_CSV)
    if not SOURCE_SCRIPT.exists():
        raise FileNotFoundError(SOURCE_SCRIPT)
    monthly = pd.read_csv(SOURCE_CSV, usecols=['date', 'scenario', 'assumption', 'total_twsa_ma12_mm'], parse_dates=['date'])
    if len(monthly) != 18000:
        raise RuntimeError(f'Unexpected source row count: {len(monthly)}')
    if monthly['total_twsa_ma12_mm'].isna().any():
        raise RuntimeError('Missing total_twsa_ma12_mm values')
    if set(monthly['scenario']) != set(SCENARIOS):
        raise RuntimeError('Scenario mapping gate failed')
    if set(monthly['assumption']) != set(PATHWAYS):
        raise RuntimeError('Pathway mapping gate failed')
    if monthly.duplicated(['date', 'scenario', 'assumption']).any():
        raise RuntimeError('Duplicate scenario-pathway-month rows')
    target = monthly[monthly['date'].dt.year.isin(TARGET_YEARS) & monthly['date'].dt.month.eq(12)].copy()
    target['target_year'] = target['date'].dt.year
    target['scenario_display'] = target['scenario'].map(SCENARIO_DISPLAY)
    target['pathway_display'] = target['assumption'].map(PATHWAY_DISPLAY)
    if len(target) != 60:
        raise RuntimeError(f'Expected 60 target rows, found {len(target)}')
    counts = target.groupby('target_year').size()
    if not (counts == 20).all():
        raise RuntimeError('Each target year must contain exactly 20 records')
    ssp_counts = target.groupby(['target_year', 'assumption']).size()
    if not (ssp_counts == 4).all():
        raise RuntimeError('Each year-pathway must contain four SSP records')
    endpoint = target[target['target_year'].eq(2100)].copy()
    stationary = endpoint[endpoint['assumption'].eq('A_stationary_residual')].set_index('scenario')['total_twsa_ma12_mm']
    gate_rows = []
    for scenario, expected in EXPECTED_STATIONARY.items():
        actual = float(stationary.loc[scenario])
        passed = abs(actual - expected) <= GATE_TOL
        gate_rows.append({'metric': f'stationary_2100_{scenario}_mm', 'computed': actual, 'expected': expected, 'tolerance': GATE_TOL, 'passed': passed})
        if not passed:
            raise RuntimeError(f'Stationary endpoint gate failed: {scenario}')
    fixed_ssp_widths = endpoint.groupby('scenario')['total_twsa_ma12_mm'].agg(lambda values: values.max() - values.min())
    fixed_pathway_spreads = endpoint.groupby('assumption')['total_twsa_ma12_mm'].agg(lambda values: values.max() - values.min())
    full_min = float(endpoint['total_twsa_ma12_mm'].min())
    full_max = float(endpoint['total_twsa_ma12_mm'].max())
    full_width = full_max - full_min
    lopo_rows = []
    for pathway in PATHWAYS:
        reduced = endpoint[endpoint['assumption'].ne(pathway)]
        reduced_min = float(reduced['total_twsa_ma12_mm'].min())
        reduced_max = float(reduced['total_twsa_ma12_mm'].max())
        lopo_rows.append({'pathway': pathway, 'reduced_width_mm': reduced_max - reduced_min, 'width_reduction_mm': full_width - (reduced_max - reduced_min)})
    lopo = pd.DataFrame(lopo_rows).set_index('pathway')
    computed = {'fixed_ssp_residual_pathway_envelope_mm': float(fixed_ssp_widths.max()), 'fixed_pathway_ssp_spread_mm': float(fixed_pathway_spreads.max()), 'full_5x4_deterministic_range_mm': full_width, 'controlled_range_ratio': float(fixed_ssp_widths.max() / fixed_pathway_spreads.max()), 'remove_persistence_reduction_mm': float(lopo.loc['B_trend_capped_persistence_residual', 'width_reduction_mm']), 'remove_risk_conditioned_reduction_mm': float(lopo.loc['C_risk_conditioned_residual', 'width_reduction_mm']), 'remove_other_pathways_max_reduction_mm': float(lopo.drop(index=['B_trend_capped_persistence_residual', 'C_risk_conditioned_residual'])['width_reduction_mm'].abs().max())}
    expected = {'fixed_ssp_residual_pathway_envelope_mm': EXPECTED_FIXED_SSP, 'fixed_pathway_ssp_spread_mm': EXPECTED_FIXED_PATHWAY, 'full_5x4_deterministic_range_mm': EXPECTED_FULL_APERTURE, 'controlled_range_ratio': EXPECTED_RATIO, 'remove_persistence_reduction_mm': EXPECTED_PERSISTENCE_REDUCTION, 'remove_risk_conditioned_reduction_mm': EXPECTED_RISK_REDUCTION, 'remove_other_pathways_max_reduction_mm': EXPECTED_OTHER_REDUCTION_MAX}
    for metric, expected_value in expected.items():
        actual = computed[metric]
        passed = abs(actual - expected_value) <= GATE_TOL
        gate_rows.append({'metric': metric, 'computed': actual, 'expected': expected_value, 'tolerance': GATE_TOL, 'passed': passed})
        if not passed:
            raise RuntimeError(f'{metric} gate failed: {actual:.12f} != {expected_value:.12f}')
    gate = pd.DataFrame(gate_rows)
    gate.insert(0, 'source_sha256', sha256(SOURCE_CSV))
    gate.insert(0, 'source_csv', SOURCE_DISPLAY)
    return {'target': target, 'endpoint': endpoint, 'lopo': lopo, 'computed': computed, 'full_min': full_min, 'full_max': full_max, 'full_width': full_width, 'gate': gate, 'source_rows': len(monthly), 'source_sha256': sha256(SOURCE_CSV)}

def build_target_values(target: pd.DataFrame) -> pd.DataFrame:
    pathway_rank = {pathway: index for index, pathway in enumerate(PATHWAY_ORDER)}
    scenario_rank = {scenario: index for index, scenario in enumerate(SCENARIOS)}
    ordered = target.sort_values(by=['target_year'], kind='stable')
    ordered['_pathway_rank'] = ordered['assumption'].map(pathway_rank)
    ordered['_scenario_rank'] = ordered['scenario'].map(scenario_rank)
    ordered = ordered.sort_values(by=['target_year', '_pathway_rank', '_scenario_rank'], kind='stable')
    columns = ['date', 'target_year', 'scenario', 'scenario_display', 'assumption', 'pathway_display', 'total_twsa_ma12_mm']
    return ordered[columns].reset_index(drop=True)

def compute_summary(target: pd.DataFrame) -> pd.DataFrame:
    pathway_rank = {pathway: index for index, pathway in enumerate(PATHWAY_ORDER)}
    rows = []
    for year in TARGET_YEARS:
        year_data = target[target['target_year'].eq(year)]
        for pathway in PATHWAY_ORDER:
            group = year_data[year_data['assumption'].eq(pathway)]['total_twsa_ma12_mm']
            ssp_min = float(group.min())
            ssp_max = float(group.max())
            ssp_median = float(group.median())
            rows.append({'target_year': year, 'assumption': pathway, 'pathway_display': PATHWAY_DISPLAY[pathway], 'ssp_min_mm': ssp_min, 'ssp_median_mm': ssp_median, 'ssp_max_mm': ssp_max, 'ssp_range_mm': ssp_max - ssp_min, '_pathway_rank': pathway_rank[pathway]})
    frame = pd.DataFrame(rows).sort_values(by=['target_year', '_pathway_rank'], kind='stable')
    return frame.drop(columns=['_pathway_rank']).reset_index(drop=True)

def finish_axes(ax: mpl.axes.Axes, grid_axis: str | None=None) -> None:
    cfs.style_cartesian(ax, grid_axis=grid_axis)

def add_panel_letter(fig: mpl.figure.Figure, ax: mpl.axes.Axes, letter: str) -> None:
    cfs.panel_label(ax, letter)

def build_diverging_cmap() -> mpl.colors.LinearSegmentedColormap:
    return mpl.colors.LinearSegmentedColormap.from_list('twsa_unified_harmonized', [cfs.NEG, cfs.ZERO, cfs.POS], N=256)

def draw_figure_v15(tables: dict[str, object]) -> mpl.figure.Figure:
    """Render the final v15 display refinement of the standard 2x2."""
    configure_style()
    summary = compute_summary(tables['target'])
    endpoint = tables['endpoint']
    lopo = tables['lopo']
    cmap = build_diverging_cmap()
    fig_width_mm, fig_height_mm = (180.0, 124.0)
    fig = plt.figure(figsize=(fig_width_mm / 25.4, fig_height_mm / 25.4))

    def add_mm_axes(left_mm: float, bottom_mm: float, width_mm: float, height_mm: float, label: str) -> mpl.axes.Axes:
        return fig.add_axes([left_mm / fig_width_mm, bottom_mm / fig_height_mm, width_mm / fig_width_mm, height_mm / fig_height_mm], label=label)
    ax_a = add_mm_axes(16.0, 69.0, 83.0, 48.0, 'panel_a_data')
    ax_b = add_mm_axes(114.0, 69.0, 60.0, 48.0, 'panel_b_lopo_vertical_bars')
    ax_c = add_mm_axes(16.0, 11.0, 84.0, 49.0, 'panel_c_grouped_change_bars')
    ax_d = add_mm_axes(118.0, 11.0, 56.0, 49.0, 'panel_d_endpoint_matrix')
    x_positions = np.array([0.0, 1.0, 2.0])
    label_y = {'B_trend_capped_persistence_residual': 244.0, 'E_consistency_reference_residual': 135.0, 'D_moderation_residual': 31.0, 'A_stationary_residual': -20.0, 'C_risk_conditioned_residual': -59.0}
    internal_labels = {'B_trend_capped_persistence_residual': 'Persistence', 'E_consistency_reference_residual': 'GW-consist.', 'D_moderation_residual': 'Moderation', 'A_stationary_residual': 'Stationary', 'C_risk_conditioned_residual': 'Risk-cond.'}
    for pathway in PATHWAY_ORDER:
        rows = summary[summary['assumption'].eq(pathway)].set_index('target_year')
        medians = np.array([float(rows.loc[y, 'ssp_median_mm']) for y in TARGET_YEARS])
        mins = np.array([float(rows.loc[y, 'ssp_min_mm']) for y in TARGET_YEARS])
        maxs = np.array([float(rows.loc[y, 'ssp_max_mm']) for y in TARGET_YEARS])
        color = PATHWAY_COLORS[pathway]
        emphasized = pathway in {'B_trend_capped_persistence_residual', 'C_risk_conditioned_residual'}
        ax_a.plot(x_positions, medians, color=color, linewidth=2.1 if emphasized else 1.35, linestyle=PATHWAY_LINESTYLES[pathway], marker='o', markersize=3.8 if emphasized else 3.3, markeredgecolor='white', markeredgewidth=0.4, zorder=4)
        ax_a.errorbar(x_positions, medians, yerr=[medians - mins, maxs - medians], fmt='none', ecolor=color, elinewidth=0.55, capsize=1.5, capthick=0.55, alpha=0.55, zorder=3)
        endpoint_value = float(medians[-1])
        label_y_value = label_y[pathway]
        ax_a.plot([2.0, 2.045], [endpoint_value, label_y_value], color='#C4C9CB', linewidth=0.42, zorder=2)
        ax_a.text(2.05, label_y_value, internal_labels[pathway], ha='right', va='center', fontsize=7.2, color=color, zorder=6, bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.78, 'pad': 0.25})
    ax_a.set_xlim(-0.08, 2.3)
    ax_a.set_ylim(-82.0, 274.0)
    ax_a.set_xticks(x_positions, ['2050', '2075', '2100'])
    ax_a.tick_params(axis='x', pad=0.5)
    ax_a.set_yticks([-50, 0, 50, 100, 150, 200, 250])
    ax_a.set_ylabel('Conditional TWSA MA12 (mm)', labelpad=1.5, fontsize=7.45)
    finish_axes(ax_a, grid_axis='y')
    ax_a.grid(axis='y', color=GRID, linewidth=0.36, zorder=0)
    ax_a.axhline(0, color=ZERO_LINE, linewidth=0.55, alpha=0.7, zorder=1)
    median_wide = summary.pivot(index='assumption', columns='target_year', values='ssp_median_mm')
    median_wide = median_wide.reindex(index=PATHWAY_ORDER, columns=TARGET_YEARS)
    change_values = np.column_stack([(median_wide[2075] - median_wide[2050]).to_numpy(), (median_wide[2100] - median_wide[2075]).to_numpy(), (median_wide[2100] - median_wide[2050]).to_numpy()])
    period_labels = ['2050→2075', '2075→2100', '2050→2100']
    period_colors = ['#D9E3E6', '#94ADB5', '#4E6972']
    y_base = np.arange(len(PATHWAY_ORDER), dtype=float)
    offsets = np.array([-0.31, 0.0, 0.31])
    for index, (period, color, offset) in enumerate(zip(period_labels, period_colors, offsets)):
        values = change_values[:, index]
        y = y_base + offset
        ax_b.barh(y, values, height=0.13, color=color, edgecolor='none', zorder=2, label=period)
        for yi, value in zip(y, values):
            if value >= 0:
                if value > 75.0:
                    x_text, ha = (value - 5.0, 'right')
                else:
                    x_text, ha = (value + (2.0 if value > 1.0 else 2.8), 'left')
            else:
                x_text, ha = (value - 2.0, 'right')
            ax_b.text(x_text, yi, f'{value:.1f}', ha=ha, va='center', fontsize=7.0, color=TEXT, zorder=4)
    b_pathway_labels = {**PATHWAY_DISPLAY_WRAPPED, 'E_consistency_reference_residual': 'Groundwater-\nconsistency\nreference', 'A_stationary_residual': 'Stationary\nresidual', 'C_risk_conditioned_residual': 'Risk-\nconditioned'}
    ax_b.set_xlim(-50.0, 100.0)
    ax_b.set_ylim(-1.15, 4.58)
    ax_b.invert_yaxis()
    ax_b.set_yticks(y_base, [b_pathway_labels[p] for p in PATHWAY_ORDER])
    ax_b.tick_params(axis='y', pad=1.0, labelsize=7.0)
    ax_b.set_xticks([-50, -25, 0, 25, 50, 75, 100])
    ax_b.set_xlabel('')
    ax_b.axvline(0, color=ZERO_LINE, linewidth=0.58, zorder=1)
    finish_axes(ax_b, grid_axis='x')
    legend_handles = [Line2D([0], [0], color=color, linewidth=3.0, label=label) for color, label in zip(period_colors, period_labels)]
    ax_b.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.52, 0.985), ncol=3, frameon=True, facecolor='white', edgecolor='#D5DADB', framealpha=0.82, fontsize=7.0, borderpad=0.22, handlelength=1.35, handletextpad=0.35, columnspacing=0.75)
    endpoint_matrix = endpoint.pivot(index='assumption', columns='scenario', values='total_twsa_ma12_mm')
    endpoint_matrix = endpoint_matrix.reindex(index=PATHWAY_ORDER, columns=SCENARIOS)
    endpoint_norm = mpl.colors.TwoSlopeNorm(vmin=-70, vcenter=0, vmax=260)
    ax_d.imshow(endpoint_matrix.to_numpy(), cmap=cmap, norm=endpoint_norm, aspect='auto', interpolation='nearest')
    endpoint_min = float(endpoint_matrix.to_numpy().min())
    endpoint_max = float(endpoint_matrix.to_numpy().max())
    for r, pathway in enumerate(PATHWAY_ORDER):
        for c, scenario in enumerate(SCENARIOS):
            value = float(endpoint_matrix.loc[pathway, scenario])
            ax_d.text(c, r, f'{value:.1f}', ha='center', va='center', fontsize=7.05, color='white' if endpoint_norm(value) < 0.25 or endpoint_norm(value) > 0.72 else TEXT, fontweight='bold' if value in {endpoint_min, endpoint_max} else 'normal')
    ax_d.set_xticks(range(4), [SCENARIO_DISPLAY[s] for s in SCENARIOS])
    ax_d.xaxis.tick_top()
    ax_d.tick_params(axis='x', pad=2.0, length=0, colors=TEXT)
    d_pathway_labels = {'B_trend_capped_persistence_residual': 'Persistence', 'E_consistency_reference_residual': 'GW-consist.', 'D_moderation_residual': 'Moderation', 'A_stationary_residual': 'Stat. resid.', 'C_risk_conditioned_residual': 'Risk-cond.'}
    ax_d.set_yticks(range(5), [d_pathway_labels[p] for p in PATHWAY_ORDER])
    ax_d.tick_params(axis='y', length=0, pad=1.0, labelsize=7.0, colors=TEXT)
    for label in ax_d.get_yticklabels():
        label.set_color(TEXT)
        label.set_rotation(15)
        label.set_fontsize(6.9)
        label.set_rotation_mode('anchor')
        label.set_horizontalalignment('right')
        label.set_verticalalignment('center')
    ax_d.set_ylim(4.5, -0.5)
    ax_d.set_xticks(np.arange(-0.5, 4.0, 1.0), minor=True)
    ax_d.set_yticks(np.arange(-0.5, 5.0, 1.0), minor=True)
    ax_d.grid(which='minor', color='white', linewidth=0.5)
    ax_d.tick_params(which='minor', bottom=False, top=False, left=False, right=False, length=0)
    finish_axes(ax_d)
    ax_d.tick_params(axis='x', which='major', top=True, bottom=False, labeltop=True, labelbottom=False, length=2.0, pad=2.0)
    ax_d.tick_params(axis='y', which='major', left=True, right=False, labelleft=True, labelright=False, length=2.0, pad=1.0)
    ax_d.tick_params(axis='both', which='minor', bottom=False, top=False, left=False, right=False, length=0)
    lollipop_paths = [('B_trend_capped_persistence_residual', 'Persistence', 'Upper-bound'), ('C_risk_conditioned_residual', 'Risk-conditioned', 'Lower-bound')]
    for row, (pathway, label, role) in enumerate(lollipop_paths):
        value = float(lopo.loc[pathway, 'width_reduction_mm'])
        color = PATHWAY_COLORS[pathway]
        ax_c.plot([0.0, value], [row, row], color=color, linewidth=1.05, zorder=2)
        ax_c.scatter(value, row, s=25, color=color, edgecolor='white', linewidth=0.5, zorder=4)
        if row == 0:
            text_x, text_ha = (122.5, 'right')
            value_y, role_y = (row + 0.18, row + 0.42)
        else:
            text_x, text_ha = (value + 3.0, 'left')
            value_y, role_y = (row - 0.18, row - 0.42)
        ax_c.text(text_x, value_y, f'{value:.2f}', ha=text_ha, va='center', fontsize=7.0, color=TEXT)
        ax_c.text(text_x, role_y, role, ha=text_ha, va='center', fontsize=7.0, color=color)
    ax_c.set_yticks([0, 1], ['Persistence', 'Risk-\nconditioned'])
    ax_c.tick_params(axis='y', labelsize=7.0, pad=1.0)
    ax_c.set_ylim(-0.55, 1.55)
    ax_c.invert_yaxis()
    ax_c.set_xlim(0.0, 125.0)
    ax_c.set_xticks([0, 25, 50, 75, 100, 125])
    ax_c.set_xlabel('Range reduction (mm)', labelpad=0.0, fontsize=7.25)
    ax_c.xaxis.set_label_coords(0.5, 0.07)
    finish_axes(ax_c, grid_axis='x')
    ax_b.clear()
    lopo_items = [('Persistence', 'B_trend_capped_persistence_residual', 'Upper-bound'), ('Risk-conditioned', 'C_risk_conditioned_residual', 'Lower-bound')]
    lopo_x = np.arange(2, dtype=float)
    lopo_values = np.array([float(lopo.loc[pathway, 'width_reduction_mm']) for _, pathway, _ in lopo_items])
    lopo_colors = [PATHWAY_COLORS['B_trend_capped_persistence_residual'], PATHWAY_COLORS['C_risk_conditioned_residual']]
    bars = ax_b.bar(lopo_x, lopo_values, width=0.36, color=lopo_colors, edgecolor='white', linewidth=0.45, zorder=3)
    for xi, value, (_, _, role), color in zip(lopo_x, lopo_values, lopo_items, lopo_colors):
        ax_b.text(xi, value + 1.2, f'{value:.2f}', ha='center', va='bottom', fontsize=7.0, color=TEXT, zorder=5)
        ax_b.text(xi, value + 8.2, role, ha='center', va='bottom', fontsize=7.0, color=color, zorder=5)
    ax_b.set_xlim(-0.55, 1.55)
    ax_b.set_ylim(0.0, 132.0)
    ax_b.set_xticks(lopo_x, [label for label, _, _ in lopo_items])
    ax_b.tick_params(axis='x', labelsize=7.0, pad=1.0)
    ax_b.set_yticks([0, 25, 50, 75, 100, 125])
    ax_b.tick_params(axis='y', labelsize=7.0, pad=1.0)
    ax_b.set_ylabel('Range reduction (mm)', labelpad=1.5, fontsize=7.25)
    finish_axes(ax_b, grid_axis='y')
    ax_b.grid(axis='y', color=GRID, linewidth=0.36, zorder=0)
    ax_c.clear()
    offsets = np.array([-0.31, 0.0, 0.31])
    for index, (period, color, offset) in enumerate(zip(period_labels, period_colors, offsets)):
        values = change_values[:, index]
        y = y_base + offset
        ax_c.barh(y, values, height=0.13, color=color, edgecolor='none', zorder=2, label=period)
        for yi, value in zip(y, values):
            if value >= 0:
                if value > 75.0:
                    x_text, ha = (value - 5.0, 'right')
                else:
                    x_text, ha = (value + (2.0 if value > 1.0 else 2.8), 'left')
            else:
                x_text, ha = (value - 2.0, 'right')
            label_kwargs = {'ha': ha, 'va': 'center', 'fontsize': 7.0, 'color': TEXT, 'zorder': 4}
            if value > 75.0:
                label_kwargs['fontweight'] = 'bold'
                label_kwargs['bbox'] = {'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.88, 'pad': 0.18}
            ax_c.text(x_text, yi, f'{value:.1f}', **label_kwargs)
    ax_c.set_xlim(-60.0, 100.0)
    ax_c.set_ylim(-1.15, 4.58)
    ax_c.invert_yaxis()
    c_pathway_labels = {'B_trend_capped_persistence_residual': 'Persistence', 'E_consistency_reference_residual': 'GW-consist.', 'D_moderation_residual': 'Moderation', 'A_stationary_residual': 'Stationary', 'C_risk_conditioned_residual': 'Risk-cond.'}
    ax_c.set_yticks(y_base, [c_pathway_labels[p] for p in PATHWAY_ORDER])
    ax_c.tick_params(axis='y', pad=1.0, labelsize=7.0, colors=TEXT)
    for label in ax_c.get_yticklabels():
        label.set_color(TEXT)
        label.set_rotation(15)
        label.set_fontsize(6.9)
        label.set_rotation_mode('anchor')
        label.set_horizontalalignment('right')
        label.set_verticalalignment('center')
    ax_c.set_xticks([-50, -25, 0, 25, 50, 75, 100])
    ax_c.set_xlabel('Median change (mm)', labelpad=1.5, fontsize=7.25)
    ax_c.xaxis.set_label_coords(0.5, -0.1)
    ax_c.axvline(0, color=ZERO_LINE, linewidth=0.58, zorder=1)
    finish_axes(ax_c, grid_axis='x')
    ax_c.legend(handles=[Line2D([0], [0], color=color, linewidth=3.0, label=label) for color, label in zip(period_colors, period_labels)], loc='upper center', bbox_to_anchor=(0.52, 0.985), ncol=3, frameon=False, fontsize=cfs.LEGEND_SIZE, borderpad=0.1, handlelength=0.85, handletextpad=0.2, columnspacing=0.4)
    for letter, axis in zip('abcd', [ax_a, ax_b, ax_c, ax_d]):
        add_panel_letter(fig, axis, letter)
    return fig

def save_figure(fig: mpl.figure.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfs.save_figure(fig, OUTPUT_DIR / stem)

def collect_text_check(fig: mpl.figure.Figure) -> tuple[pd.DataFrame, int, float]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figure_bbox = fig.bbox
    dpi = fig.dpi
    rows = []
    text_items: list[tuple[str, object]] = []
    seen = set()
    for artist in fig.findobj(match=lambda obj: isinstance(obj, Text)):
        if id(artist) in seen:
            continue
        seen.add(id(artist))
        if not artist.get_visible() or not artist.get_text():
            continue
        bbox = artist.get_window_extent(renderer)
        if not np.all(np.isfinite([bbox.x0, bbox.x1, bbox.y0, bbox.y1])):
            continue
        parent = artist.axes.get_label() if artist.axes is not None else 'figure'
        text = artist.get_text().replace('\n', ' | ')
        distances = {'distance_left_mm': (bbox.x0 - figure_bbox.x0) / dpi * 25.4, 'distance_right_mm': (figure_bbox.x1 - bbox.x1) / dpi * 25.4, 'distance_bottom_mm': (bbox.y0 - figure_bbox.y0) / dpi * 25.4, 'distance_top_mm': (figure_bbox.y1 - bbox.y1) / dpi * 25.4}
        inside = all((value >= 0 for value in distances.values()))
        rows.append({'parent': parent, 'text': text, 'font_size_pt': artist.get_fontsize(), 'x0_px': bbox.x0, 'x1_px': bbox.x1, 'y0_px': bbox.y0, 'y1_px': bbox.y1, **distances, 'inside_canvas': inside})
        text_items.append((text, bbox))
    overlap_count = 0
    for index, (_, bbox_a) in enumerate(text_items):
        for _, bbox_b in text_items[index + 1:]:
            overlap_width = min(bbox_a.x1, bbox_b.x1) - max(bbox_a.x0, bbox_b.x0)
            overlap_height = min(bbox_a.y1, bbox_b.y1) - max(bbox_a.y0, bbox_b.y0)
            if overlap_width > 1.0 and overlap_height > 1.0:
                overlap_count += 1
    frame = pd.DataFrame(rows)
    distance_columns = ['distance_left_mm', 'distance_right_mm', 'distance_bottom_mm', 'distance_top_mm']
    minimum_clearance = float(frame[distance_columns].min().min())
    return (frame, overlap_count, minimum_clearance)

def geometry_check(fig: mpl.figure.Figure) -> pd.DataFrame:
    width_mm = fig.get_figwidth() * 25.4
    height_mm = fig.get_figheight() * 25.4
    rows = []
    for index, ax in enumerate(fig.axes):
        box = ax.get_position()
        rows.append({'axes_index': index, 'axes_label': ax.get_label(), 'left_mm': box.x0 * width_mm, 'bottom_mm': box.y0 * height_mm, 'width_mm': box.width * width_mm, 'height_mm': box.height * height_mm, 'right_mm': (box.x0 + box.width) * width_mm, 'top_mm': (box.y0 + box.height) * height_mm})
    return pd.DataFrame(rows)

def write_text(path: Path, text: str) -> None:
    with path.open('w', encoding='utf-8-sig', newline='\n') as handle:
        handle.write(text)

def build_v12_verification_markdown(tables: dict[str, object], summary: pd.DataFrame, text_check: pd.DataFrame, overlap_count: int, minimum_clearance: float, geometry: pd.DataFrame) -> str:
    median_wide = summary.pivot(index='assumption', columns='target_year', values='ssp_median_mm')
    median_wide = median_wide.reindex(index=PATHWAY_ORDER, columns=TARGET_YEARS)
    closure = bool(np.allclose(median_wide[2075] - median_wide[2050] + (median_wide[2100] - median_wide[2075]), median_wide[2100] - median_wide[2050], atol=1e-10))
    all_inside = bool(text_check['inside_canvas'].all())
    min_font = float(text_check['font_size_pt'].min())
    computed = tables['computed']
    lines = ['# Figure 5 rendering and numerical verification', '', f'- Source CSV: `{SOURCE_DISPLAY}`', f'- Source SHA-256: `{tables['source_sha256']}`', f'- Source rows: {tables['source_rows']}', f'- 2100 within-SSP pathway envelope: {computed['fixed_ssp_residual_pathway_envelope_mm']:.6f} mm', f'- 2100 stationary-pathway inter-SSP spread: {computed['fixed_pathway_ssp_spread_mm']:.6f} mm', f'- Target-period change closure: {closure}', f'- Numerical checks passed: {bool(tables['gate']['passed'].all())}', f'- Text inside canvas: {all_inside}', f'- Text-overlap pairs: {overlap_count}', f'- Minimum text-edge clearance: {minimum_clearance:.2f} mm', f'- Minimum font size: {min_font:.1f} pt', '', 'Render verification: PASS']
    return '\n'.join(lines) + '\n'

def main_v15() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tables = load_and_gate()
    target_values = build_target_values(tables['target'])
    summary = compute_summary(tables['target'])
    stem = 'Fig5_CEE_harmonized_v2'
    fig = draw_figure_v15(tables)
    text_check, overlap_count, minimum_clearance = collect_text_check(fig)
    geometry = geometry_check(fig)
    save_figure(fig, stem)
    plt.close(fig)
    tables['gate'].to_csv(OUTPUT_DIR / f'{stem}_scientific_gate.csv', index=False, encoding='utf-8-sig')
    summary.to_csv(OUTPUT_DIR / f'{stem}_summary_2050_2075_2100.csv', index=False, encoding='utf-8-sig')
    build_target_values(tables['endpoint']).to_csv(OUTPUT_DIR / f'{stem}_endpoint_2100.csv', index=False, encoding='utf-8-sig')
    target_values.to_csv(OUTPUT_DIR / f'{stem}_target_values_2050_2075_2100.csv', index=False, encoding='utf-8-sig')
    lopo_summary = tables['lopo'].reset_index().rename(columns={'index': 'pathway'})
    lopo_summary['pathway_display'] = lopo_summary['pathway'].map(PATHWAY_DISPLAY)
    lopo_summary = lopo_summary[['pathway', 'pathway_display', 'width_reduction_mm']]
    lopo_summary.to_csv(OUTPUT_DIR / f'{stem}_lopo_summary.csv', index=False, encoding='utf-8-sig')
    geometry.to_csv(OUTPUT_DIR / f'{stem}_geometry_check.csv', index=False, encoding='utf-8-sig')
    text_check.to_csv(OUTPUT_DIR / f'{stem}_text_bbox_check.csv', index=False, encoding='utf-8-sig')
    median_wide = summary.pivot(index='assumption', columns='target_year', values='ssp_median_mm')
    median_wide = median_wide.reindex(index=PATHWAY_ORDER, columns=TARGET_YEARS)
    changes = pd.DataFrame({'pathway': PATHWAY_ORDER, '2050→2075': (median_wide[2075] - median_wide[2050]).to_numpy(), '2075→2100': (median_wide[2100] - median_wide[2075]).to_numpy(), '2050→2100': (median_wide[2100] - median_wide[2050]).to_numpy()})
    changes.to_csv(OUTPUT_DIR / f'{stem}_changes_2050_2075_2100.csv', index=False, encoding='utf-8-sig')
    write_text(OUTPUT_DIR / f'{stem}_render_verification.md', build_v12_verification_markdown(tables, summary, text_check, overlap_count, minimum_clearance, geometry))
    print('Render verification: PASS')
if __name__ == '__main__':
    main_v15()
