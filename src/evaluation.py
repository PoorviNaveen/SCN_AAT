"""
Evaluation metrics and comparative analysis ? Sections 7 & 8 of the paper.
Computes RMSE, R2, MAPE, MABE and generates all comparison figures.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# Metric helpers
# -----------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, label=''):
    """Return dict with R2, RMSE, MAPE, MABE."""
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mabe = mean_absolute_error(y_true, y_pred)
    # Avoid division by zero in MAPE
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    print(f"\n  -- {label} --")
    print(f"     R2   : {r2:.4f}")
    print(f"     RMSE : {rmse:.5f}")
    print(f"     MAPE : {mape:.2f} %")
    print(f"     MABE : {mabe:.5f}")

    return {'model': label, 'R2': r2, 'RMSE': rmse, 'MAPE': mape, 'MABE': mabe}


# -----------------------------------------------------------------------------
# Scatter plot (predicted vs actual)  ? matches Figs 15?16, 19?20 of paper
# -----------------------------------------------------------------------------

def plot_scatter(y_true, y_pred, r2, title, filename):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, alpha=0.35, s=10, color='steelblue', label='Data points')
    lim = [min(y_true.min(), y_pred.min()) - 0.02,
           max(y_true.max(), y_pred.max()) + 0.02]
    ax.plot(lim, lim, 'r--', lw=1.5, label='Perfect fit')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('Actual Traffic Volume (normalised)')
    ax.set_ylabel('Predicted Traffic Volume (normalised)')
    ax.set_title(f'{title}\n$R^2$ = {r2:.4f}')
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {path}")


# -----------------------------------------------------------------------------
# Training RMSE history
# -----------------------------------------------------------------------------

def plot_training_curve(anfis_rmse_history, title='ANFIS Training RMSE', filename='anfis_training_curve.png'):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(anfis_rmse_history, color='royalblue', lw=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {path}")


# -----------------------------------------------------------------------------
# GA evolution curve
# -----------------------------------------------------------------------------

def plot_ga_evolution(logbook, filename='anfis_ga_evolution.png'):
    gens = [r['gen']  for r in logbook]
    mins = [r['min']  for r in logbook]
    avgs = [r['avg']  for r in logbook]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(gens, mins, label='Best RMSE',    color='green',  lw=1.5)
    ax.plot(gens, avgs, label='Average RMSE', color='orange', lw=1.2, linestyle='--')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness (RMSE)')
    ax.set_title('ANFIS-GA: Genetic Algorithm Evolution')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {path}")


# -----------------------------------------------------------------------------
# Comprehensive comparative dashboard ? Section 7
# -----------------------------------------------------------------------------

def plot_comparison_dashboard(results: dict, filename='comparison_dashboard.png'):
    """
    results = {
        'ANFIS':    {'train': metrics_dict, 'test': metrics_dict},
        'ANFIS-GA': {'train': metrics_dict, 'test': metrics_dict},
    }
    """
    models  = list(results.keys())
    metrics = ['R2', 'RMSE', 'MAPE', 'MABE']
    colors  = ['#2196F3', '#4CAF50']   # blue, green

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Comparative Analysis: ANFIS vs ANFIS-GA\n(Sections 7.1?7.6)',
                 fontsize=14, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.38)

    for col, metric in enumerate(metrics):
        # -- Training bar ------------------------------------------------------
        ax_tr = fig.add_subplot(gs[0, col])
        vals  = [results[m]['train'][metric] for m in models]
        bars  = ax_tr.bar(models, vals, color=colors, edgecolor='black', linewidth=0.6)
        ax_tr.set_title(f'Training {metric}', fontsize=10)
        ax_tr.set_ylim(0, max(vals) * 1.25 + 1e-6)
        for bar, val in zip(bars, vals):
            ax_tr.text(bar.get_x() + bar.get_width() / 2,
                       bar.get_height() + max(vals) * 0.03,
                       f'{val:.4f}', ha='center', fontsize=8)
        ax_tr.grid(axis='y', linestyle='--', alpha=0.5)

        # -- Testing bar -------------------------------------------------------
        ax_te = fig.add_subplot(gs[1, col])
        vals  = [results[m]['test'][metric] for m in models]
        bars  = ax_te.bar(models, vals, color=colors, edgecolor='black', linewidth=0.6)
        ax_te.set_title(f'Testing {metric}', fontsize=10)
        ax_te.set_ylim(0, max(vals) * 1.25 + 1e-6)
        for bar, val in zip(bars, vals):
            ax_te.text(bar.get_x() + bar.get_width() / 2,
                       bar.get_height() + max(vals) * 0.03,
                       f'{val:.4f}', ha='center', fontsize=8)
        ax_te.grid(axis='y', linestyle='--', alpha=0.5)

    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {path}")


# -----------------------------------------------------------------------------
# Actual vs Predicted time-series plot ? Section 8.1 Output Analysis
# -----------------------------------------------------------------------------

def plot_actual_vs_predicted(y_true, y_pred_anfis, y_pred_ga,
                              n_samples=200, filename='actual_vs_predicted.png'):
    idx = np.arange(n_samples)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(idx, y_true[:n_samples], label='Actual',       color='black',  lw=1.2)
    axes[0].plot(idx, y_pred_anfis[:n_samples], label='ANFIS',  color='blue',   lw=1.0, linestyle='--')
    axes[0].set_title('ANFIS: Actual vs Predicted Traffic Volume (Test Set)')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].set_ylabel('Volume (normalised)')

    axes[1].plot(idx, y_true[:n_samples], label='Actual',        color='black',  lw=1.2)
    axes[1].plot(idx, y_pred_ga[:n_samples],   label='ANFIS-GA', color='green',  lw=1.0, linestyle='--')
    axes[1].set_title('ANFIS-GA: Actual vs Predicted Traffic Volume (Test Set)')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[1].set_ylabel('Volume (normalised)')
    axes[1].set_xlabel('Sample index')

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {path}")


# -----------------------------------------------------------------------------
# Summary metrics table
# -----------------------------------------------------------------------------

def print_summary_table(results):
    """Print Section 7 comparative table."""
    print("\n" + "=" * 70)
    print("  COMPARATIVE ANALYSIS SUMMARY  (Sections 7.1 ? 7.6)")
    print("=" * 70)
    print(f"  {'Metric':<12} {'ANFIS Train':>14} {'ANFIS Test':>12} "
          f"{'ANFIS-GA Train':>16} {'ANFIS-GA Test':>14}")
    print("-" * 70)
    for m in ['R2', 'RMSE', 'MAPE', 'MABE']:
        at  = results['ANFIS']['train'][m]
        ae  = results['ANFIS']['test'][m]
        gt  = results['ANFIS-GA']['train'][m]
        ge  = results['ANFIS-GA']['test'][m]
        print(f"  {m:<12} {at:>14.4f} {ae:>12.4f} {gt:>16.4f} {ge:>14.4f}")
    print("=" * 70)
    print("  Paper targets: ANFIS test R2=0.9790 | ANFIS-GA test R2=0.9980")
    print("=" * 70)


def print_cross_split_table(results_70, results_80):
    """Print complete side-by-side comparative table for 70:30 vs 80:20 splits."""
    print("\n" + "=" * 84)
    print("  CROSS-SPLIT BENCHMARK SUMMARY: 70:30 VS 80:20 TRAIN-TEST SPLIT")
    print("=" * 84)
    print(f"  {'Metric / Model':<20} {'70:30 (Train)':>14} {'70:30 (Test)':>14} {'80:20 (Train)':>14} {'80:20 (Test)':>14}")
    print("-" * 84)

    for model in ['ANFIS', 'ANFIS-GA']:
        print(f"  [{model}]")
        for m in ['R2', 'RMSE', 'MAPE', 'MABE']:
            v70_tr = results_70[model]['train'][m]
            v70_te = results_70[model]['test'][m]
            v80_tr = results_80[model]['train'][m]
            v80_te = results_80[model]['test'][m]
            unit = '%' if m == 'MAPE' else ''
            print(f"    {m:<16} {v70_tr:>13.4f}{unit} {v70_te:>13.4f}{unit} {v80_tr:>13.4f}{unit} {v80_te:>13.4f}{unit}")
        t70 = results_70[model]['train_time']
        t80 = results_80[model]['train_time']
        print(f"    {'Runtime':<16} {t70:>13.1f}s {'-':>14} {t80:>13.1f}s {'-':>14}")
        print("-" * 84)
    print("=" * 84)


def plot_split_comparison_dashboard(results_70, results_80, filename='split_comparison_dashboard.png'):
    """
    Visual comparison dashboard comparing 70:30 split vs 80:20 split
    for both ANFIS and ANFIS-GA across test R2, RMSE, MABE, and Runtime.
    """
    metrics = [
        ('R2', 'Test R^2 Score (Higher is Better)', True),
        ('RMSE', 'Test RMSE (Lower is Better)', False),
        ('MABE', 'Test MABE (Lower is Better)', False),
        ('MAPE', 'Test MAPE (%) (Lower is Better)', False)
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Cross-Split Performance Evaluation: 70:30 vs 80:20 Train-Test Splits\n(Standalone ANFIS vs Hybrid ANFIS-GA)',
                 fontsize=14, fontweight='bold', y=0.98, color='#1B365D')

    models = ['ANFIS', 'ANFIS-GA']
    x = np.arange(len(models))
    width = 0.32

    colors_70 = '#3498DB'
    colors_80 = '#E67E22'

    for idx, (m_key, title, higher_is_better) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        vals_70 = [results_70[m]['test'][m_key] for m in models]
        vals_80 = [results_80[m]['test'][m_key] for m in models]

        rects1 = ax.bar(x - width/2, vals_70, width, label='70:30 Split (595 / 255)', color=colors_70, edgecolor='#2980B9', linewidth=1.2)
        rects2 = ax.bar(x + width/2, vals_80, width, label='80:20 Split (680 / 170)', color=colors_80, edgecolor='#D35400', linewidth=1.2)

        ax.set_title(title, fontsize=11, fontweight='bold', color='#2C3E50', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(['Standalone ANFIS', 'Hybrid ANFIS-GA'], fontsize=10, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        max_val = max(max(vals_70), max(vals_80))
        ax.set_ylim(0, max_val * 1.25 + 1e-4)

        for rect in rects1:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., h + max_val * 0.02,
                    f'{h:.4f}' if m_key != 'MAPE' else f'{h:.2f}%',
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1B4F72')

        for rect in rects2:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., h + max_val * 0.02,
                    f'{h:.4f}' if m_key != 'MAPE' else f'{h:.2f}%',
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#935116')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {path}")


# -----------------------------------------------------------------------------
# Research Paper vs Experimental Benchmark Comparison
# -----------------------------------------------------------------------------

def print_paper_comparison(results_70, results_80):
    """
    Prints a detailed 4-way experimental comparison against the published Research Paper:
    Olayode et al., Applied Sciences, 2023, 13, 744.
    """
    print("\n" + "=" * 96)
    print("  COMPREHENSIVE BENCHMARK: ALL 4 EXPERIMENTAL RESULTS vs RESEARCH PAPER (Olayode et al., 2023)")
    print("=" * 96)
    print(f"  {'Configuration':<26} {'Phase':<8} {'R2':>8} {'RMSE':>10} {'MAPE (%)':>10} {'MABE':>10} {'Status':>16}")
    print("-" * 96)

    # Published Research Paper (70:30 baseline)
    print(f"  {'Paper ANFIS (70:30)':<26} {'Train':<8} {'0.9709':>8} {'-':>10} {'-':>10} {'-':>10} {'Published':>16}")
    print(f"  {'Paper ANFIS (70:30)':<26} {'Test':<8} {'0.9790':>8} {'-':>10} {'-':>10} {'-':>10} {'Published':>16}")
    print(f"  {'Paper ANFIS-GA (70:30)':<26} {'Train':<8} {'0.8979':>8} {'-':>10} {'-':>10} {'-':>10} {'Published':>16}")
    print(f"  {'Paper ANFIS-GA (70:30)':<26} {'Test':<8} {'0.9980':>8} {'-':>10} {'-':>10} {'-':>10} {'Published Target':>16}")
    print("-" * 96)

    # Implemented 4 results
    exp_configs = [
        ('ANFIS (70:30 Split)', results_70['ANFIS']),
        ('ANFIS-GA (70:30 Split)', results_70['ANFIS-GA']),
        ('ANFIS (80:20 Split)', results_80['ANFIS']),
        ('ANFIS-GA (80:20 Split)', results_80['ANFIS-GA']),
    ]
    for name, data in exp_configs:
        for phase in ['train', 'test']:
            r2 = f"{data[phase]['R2']:.4f}"
            rmse = f"{data[phase]['RMSE']:.4f}"
            mape = f"{data[phase]['MAPE']:.2f}%"
            mabe = f"{data[phase]['MABE']:.4f}"
            status = 'Leak-Free (Tr)' if phase == 'train' else 'Leak-Free (Te)'
            print(f"  {name:<26} {phase.capitalize():<8} {r2:>8} {rmse:>10} {mape:>10} {mabe:>10} {status:>16}")
        print("-" * 96)
    print("=" * 96)


def save_paper_comparison_csv(results_70, results_80, filename='paper_vs_experimental_comparison.csv'):
    """Save complete comparison between paper baseline and all 4 experimental results."""
    import pandas as pd
    rows = [
        # Paper targets
        {'Source': 'Research Paper (Olayode et al., 2023)', 'Split': '70:30', 'Model': 'ANFIS', 'Phase': 'Train', 'R2': 0.9709, 'RMSE': None, 'MAPE': None, 'MABE': None},
        {'Source': 'Research Paper (Olayode et al., 2023)', 'Split': '70:30', 'Model': 'ANFIS', 'Phase': 'Test', 'R2': 0.9790, 'RMSE': None, 'MAPE': None, 'MABE': None},
        {'Source': 'Research Paper (Olayode et al., 2023)', 'Split': '70:30', 'Model': 'ANFIS-GA', 'Phase': 'Train', 'R2': 0.8979, 'RMSE': None, 'MAPE': None, 'MABE': None},
        {'Source': 'Research Paper (Olayode et al., 2023)', 'Split': '70:30', 'Model': 'ANFIS-GA', 'Phase': 'Test', 'R2': 0.9980, 'RMSE': None, 'MAPE': None, 'MABE': None},
        {'Source': 'Research Paper (Olayode et al., 2023)', 'Split': '80:20', 'Model': 'ANFIS', 'Phase': 'Not Tested In Paper', 'R2': None, 'RMSE': None, 'MAPE': None, 'MABE': None},
        {'Source': 'Research Paper (Olayode et al., 2023)', 'Split': '80:20', 'Model': 'ANFIS-GA', 'Phase': 'Not Tested In Paper', 'R2': None, 'RMSE': None, 'MAPE': None, 'MABE': None},
    ]
    # Experimental rows
    for split_tag, res in [('70:30', results_70), ('80:20', results_80)]:
        for model in ['ANFIS', 'ANFIS-GA']:
            for phase in ['train', 'test']:
                m = res[model][phase]
                rows.append({
                    'Source': 'Our Implementation (Leak-Free)',
                    'Split': split_tag,
                    'Model': model,
                    'Phase': phase.capitalize(),
                    'R2': m['R2'],
                    'RMSE': m['RMSE'],
                    'MAPE': m['MAPE'],
                    'MABE': m['MABE']
                })
    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, filename)
    df.to_csv(path, index=False)
    print(f"  [Saved] {path}")
    return df


def plot_paper_comparison_barchart(results_70, results_80, filename='paper_vs_experimental_comparison.png'):
    """
    Visual comparison chart showing Test R2 for Paper vs 70:30 vs 80:20 experimental splits.
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

    categories = ['Paper (70:30)', 'Experiment (70:30)', 'Experiment (80:20)']
    anfis_r2 = [0.9790, results_70['ANFIS']['test']['R2'], results_80['ANFIS']['test']['R2']]
    anfis_ga_r2 = [0.9980, results_70['ANFIS-GA']['test']['R2'], results_80['ANFIS-GA']['test']['R2']]

    x = np.arange(len(categories))
    width = 0.35

    rects1 = ax.bar(x - width/2, anfis_r2, width, label='Standalone ANFIS', color='#3498DB', edgecolor='#1F618D', linewidth=1.2)
    rects2 = ax.bar(x + width/2, anfis_ga_r2, width, label='Hybrid ANFIS-GA', color='#27AE60', edgecolor='#196F3D', linewidth=1.2)

    ax.set_ylabel('Test $R^2$ Score (Higher is Better)', fontsize=12, fontweight='bold', color='#1B365D')
    ax.set_title('Test $R^2$ Benchmark: Research Paper vs Experimental Dual-Split Implementation', fontsize=13, fontweight='bold', pad=15, color='#1B365D')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_ylim(0.70, 1.05)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for rect in rects1:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., h + 0.008, f'{h:.4f}', ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#1F618D')

    for rect in rects2:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., h + 0.008, f'{h:.4f}', ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#196F3D')

    # Draw improvement callouts
    for i in range(len(categories)):
        gain = anfis_ga_r2[i] - anfis_r2[i]
        pct = (gain / anfis_r2[i]) * 100
        mid_x = x[i]
        top_y = max(anfis_r2[i], anfis_ga_r2[i]) + 0.025
        ax.text(mid_x, top_y, f'+{gain:.4f} (hybrid gain)', ha='center', va='bottom', fontsize=8.5, fontstyle='italic', color='#884EA0', fontweight='bold')

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {path}")
