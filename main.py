"""
Main runner -- trains ANFIS and ANFIS-GA under both 70:30 and 80:20 splits,
evaluates performance, conducts cross-split comparative analysis, and saves all results.
Run:  python -u main.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import time
import pandas as pd

from src.data_preprocessing import load_and_preprocess
from src.anfis_model        import ANFIS
from src.anfis_ga_model     import ANFIS_GA
from src.evaluation         import (
    compute_metrics,
    plot_scatter,
    plot_training_curve,
    plot_ga_evolution,
    plot_comparison_dashboard,
    plot_actual_vs_predicted,
    print_summary_table,
    print_cross_split_table,
    plot_split_comparison_dashboard,
    print_paper_comparison,
    save_paper_comparison_csv,
    plot_paper_comparison_barchart,
)

RANDOM_STATE = 42


def run_pipeline_for_split(train_split=0.70, split_tag="70_30"):
    """
    Executes end-to-end training and evaluation for a given train-test split.
    train_split: 0.70 for 70:30, 0.80 for 80:20
    """
    train_pct = int(round(train_split * 100))
    test_pct  = 100 - train_pct
    print("\n" + "=" * 65)
    print(f"  RUNNING PIPELINE: {train_pct}:{test_pct} TRAIN-TEST SPLIT")
    print("=" * 65)
    sys.stdout.flush()

    # Load data for this split
    X_train, X_test, y_train, y_test, scaler_X, scaler_y, df = load_and_preprocess(
        n_samples=850, random_state=RANDOM_STATE, train_split=train_split
    )

    n_inputs = X_train.shape[1]
    n_mfs = 2 if n_inputs >= 3 else 3
    results = {}

    # -------------------------------------------------------------------------
    # 1. Non-Hybrid ANFIS
    # -------------------------------------------------------------------------
    print(f"\n>>> [{split_tag}] Standalone ANFIS Training ({n_inputs} inputs, {n_mfs} MFs = {n_mfs**n_inputs} rules)")
    sys.stdout.flush()

    anfis = ANFIS(
        n_inputs=n_inputs,
        n_mfs=n_mfs,
        epochs=100,
        random_state=RANDOM_STATE
    )

    t0 = time.time()
    anfis.fit(X_train, y_train)
    anfis_train_time = time.time() - t0
    print(f"\n  Training time: {anfis_train_time:.1f}s")
    sys.stdout.flush()

    y_pred_anfis_train = anfis.predict(X_train)
    y_pred_anfis_test  = anfis.predict(X_test)

    m_tr = compute_metrics(y_train, y_pred_anfis_train, f'ANFIS Training ({split_tag})')
    m_te = compute_metrics(y_test,  y_pred_anfis_test,  f'ANFIS Testing ({split_tag})')
    results['ANFIS'] = {
        'train': m_tr,
        'test': m_te,
        'train_time': anfis_train_time,
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }

    plot_training_curve(
        anfis.train_rmse_history,
        title=f'ANFIS Training RMSE Convergence ({train_pct}:{test_pct} Split)',
        filename=f'anfis_training_curve_{split_tag}.png'
    )
    plot_scatter(
        y_train, y_pred_anfis_train, m_tr['R2'],
        f'ANFIS Training ({train_pct}:{test_pct} Split)',
        f'anfis_train_scatter_{split_tag}.png'
    )
    plot_scatter(
        y_test, y_pred_anfis_test, m_te['R2'],
        f'ANFIS Testing ({train_pct}:{test_pct} Split)',
        f'anfis_test_scatter_{split_tag}.png'
    )

    # -------------------------------------------------------------------------
    # 2. Hybrid ANFIS-GA
    # -------------------------------------------------------------------------
    print(f"\n>>> [{split_tag}] Hybrid ANFIS-GA Training")
    sys.stdout.flush()

    anfis_ga = ANFIS_GA(
        n_inputs=n_inputs,
        n_mfs=n_mfs,
        pop_size=60 if n_inputs >= 3 else 100,
        n_generations=200 if n_inputs >= 3 else 500,
        cx_prob=0.80,
        mut_prob=0.30,
        mut_sigma=0.02,
        tournament_size=8,
        epochs_anfis=50,
        random_state=RANDOM_STATE
    )

    t0 = time.time()
    anfis_ga.fit(X_train, y_train)
    ga_train_time = time.time() - t0
    print(f"\n  Training time: {ga_train_time:.1f}s")
    sys.stdout.flush()

    y_pred_ga_train = anfis_ga.predict(X_train)
    y_pred_ga_test  = anfis_ga.predict(X_test)

    m_tr_ga = compute_metrics(y_train, y_pred_ga_train, f'ANFIS-GA Training ({split_tag})')
    m_te_ga = compute_metrics(y_test,  y_pred_ga_test,  f'ANFIS-GA Testing ({split_tag})')
    results['ANFIS-GA'] = {
        'train': m_tr_ga,
        'test': m_te_ga,
        'train_time': ga_train_time,
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }

    plot_ga_evolution(
        anfis_ga.ga_logbook,
        filename=f'anfis_ga_evolution_{split_tag}.png'
    )
    plot_scatter(
        y_train, y_pred_ga_train, m_tr_ga['R2'],
        f'ANFIS-GA Training ({train_pct}:{test_pct} Split)',
        f'anfis_ga_train_scatter_{split_tag}.png'
    )
    plot_scatter(
        y_test, y_pred_ga_test, m_te_ga['R2'],
        f'ANFIS-GA Testing ({train_pct}:{test_pct} Split)',
        f'anfis_ga_test_scatter_{split_tag}.png'
    )

    # -------------------------------------------------------------------------
    # Dashboard and Waveform for this split
    # -------------------------------------------------------------------------
    plot_comparison_dashboard(results, filename=f'comparison_dashboard_{split_tag}.png')
    plot_actual_vs_predicted(
        y_test, y_pred_anfis_test, y_pred_ga_test,
        n_samples=min(100, len(y_test)),
        filename=f'actual_vs_predicted_{split_tag}.png'
    )

    # Standard alias plots (for primary 70:30 split compatibility)
    if split_tag == "70_30":
        plot_training_curve(anfis.train_rmse_history, filename='anfis_training_curve.png')
        plot_scatter(y_train, y_pred_anfis_train, m_tr['R2'], 'ANFIS Model Training', 'anfis_train_scatter.png')
        plot_scatter(y_test,  y_pred_anfis_test,  m_te['R2'], 'ANFIS Model Testing',  'anfis_test_scatter.png')
        plot_ga_evolution(anfis_ga.ga_logbook, filename='anfis_ga_evolution.png')
        plot_scatter(y_train, y_pred_ga_train, m_tr_ga['R2'], 'ANFIS-GA Model Training', 'anfis_ga_train_scatter.png')
        plot_scatter(y_test,  y_pred_ga_test,  m_te_ga['R2'], 'ANFIS-GA Model Testing',  'anfis_ga_test_scatter.png')
        plot_comparison_dashboard(results, filename='comparison_dashboard.png')
        plot_actual_vs_predicted(y_test, y_pred_anfis_test, y_pred_ga_test, n_samples=min(100, len(y_test)), filename='actual_vs_predicted.png')

    print_summary_table(results)
    return results


def main():
    print("\n" + "#" * 65)
    print("  ANFIS vs ANFIS-GA -- DUAL-SPLIT COMPARATIVE BENCHMARK")
    print("  Evaluating 70:30 vs 80:20 Train-Test Partitions (850 Samples)")
    print("#" * 65)
    sys.stdout.flush()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # =========================================================================
    # PART 1: 70:30 Split Benchmark (595 Train / 255 Test)
    # =========================================================================
    results_70 = run_pipeline_for_split(train_split=0.70, split_tag="70_30")

    # =========================================================================
    # PART 2: 80:20 Split Benchmark (680 Train / 170 Test)
    # =========================================================================
    results_80 = run_pipeline_for_split(train_split=0.80, split_tag="80_20")

    # =========================================================================
    # PART 3: Cross-Split Comparative Analysis & Visualizations
    # =========================================================================
    print_cross_split_table(results_70, results_80)
    plot_split_comparison_dashboard(results_70, results_80, filename='split_comparison_dashboard.png')

    # Compare all 4 experimental results with published research paper results
    print_paper_comparison(results_70, results_80)
    save_paper_comparison_csv(results_70, results_80, filename='paper_vs_experimental_comparison.csv')
    plot_paper_comparison_barchart(results_70, results_80, filename='paper_vs_experimental_comparison.png')

    # Save detailed CSV with both splits
    csv_rows = []
    for split_tag, res_dict in [('70:30', results_70), ('80:20', results_80)]:
        for model_name, m_data in res_dict.items():
            for phase in ['train', 'test']:
                row = {
                    'Split': split_tag,
                    'Model': model_name,
                    'Phase': phase,
                    'TrainSamples': m_data['train_samples'],
                    'TestSamples': m_data['test_samples'],
                    'R2': m_data[phase]['R2'],
                    'RMSE': m_data[phase]['RMSE'],
                    'MAPE': m_data[phase]['MAPE'],
                    'MABE': m_data[phase]['MABE'],
                    'TrainingTime_s': m_data['train_time']
                }
                csv_rows.append(row)

    df_metrics = pd.DataFrame(csv_rows)
    metrics_path = os.path.join(RESULTS_DIR, 'split_comparison_metrics.csv')
    df_metrics.to_csv(metrics_path, index=False)
    print(f"\n  [Saved] {metrics_path}")

    # Save standard metrics_summary.csv (matching the 70:30 benchmark)
    summary_path = os.path.join(RESULTS_DIR, 'metrics_summary.csv')
    df_metrics[df_metrics['Split'] == '70:30'].to_csv(summary_path, index=False)
    print(f"  [Saved] {summary_path}")

    print("\n" + "#" * 65)
    print("  Dual-Split Comparison Completed Successfully!")
    print("  All cross-split plots and metrics saved to results/")
    print("#" * 65 + "\n")
    sys.stdout.flush()


if __name__ == '__main__':
    main()
