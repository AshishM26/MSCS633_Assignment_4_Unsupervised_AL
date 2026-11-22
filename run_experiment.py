"""Run the unsupervised PyOD credit-card fraud benchmark."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fraud_detection.config import (BATCH_SIZE, CONTAMINATION, DATASET_PATH,
                                    EPOCHS, RANDOM_STATE, ROOT)
from fraud_detection.data import load_data, split_data
from fraud_detection.evaluation import evaluate
from fraud_detection.model import build_model, select_device
from fraud_detection.plotting import save_plots
from fraud_detection.preprocessing import scale_features


def positive_int(value):
    """Parse a strictly positive integer CLI argument."""
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return result


def output_directory(quick):
    """Separate development results from full benchmark artifacts."""
    path = ROOT / 'outputs' / ('quick' if quick else '')
    path.mkdir(parents=True, exist_ok=True)
    return path


def run(args):
    """Train and evaluate the primary model and optional reference model."""
    start = time.perf_counter()
    device = select_device(args.device)
    frame = load_data(args.data)
    counts = frame['Class'].value_counts()
    out = output_directory(args.quick)
    if args.quick:
        frame, _ = train_test_split(frame, train_size=20000, stratify=frame['Class'],
                                    random_state=RANDOM_STATE)
    x_train, x_test, y_train, y_test = split_data(frame)
    train, test = scale_features(x_train, x_test)
    info = {'dataset': {'rows': int(sum(counts)), 'columns': 31,
                        'legitimate': int(counts[0]), 'fraud': int(counts[1]),
                        'fraud_percentage': float(counts[1] / sum(counts) * 100)},
            'quick': args.quick, 'train_rows': len(train), 'test_rows': len(test),
            'test_fraud': int(y_test.sum()), 'device': device,
            'configuration': {'contamination': CONTAMINATION, 'preprocessing': False,
                              'hidden_neuron_list': [64, 32], 'hidden_activation_name': 'relu',
                              'batch_norm': True, 'dropout_rate': 0.2, 'lr': 0.001,
                              'epoch_num': args.epochs, 'batch_size': args.batch_size,
                              'random_state': RANDOM_STATE}}
    print('=' * 60, '\n PyOD AutoEncoder Fraud Detection\n MSCS-633 Hands-On Assignment 4',
          '\n' + '=' * 60, flush=True)
    if args.quick:
        print('QUICK / DEVELOPMENT MODE — NOT FINAL RESULTS', flush=True)
    print(f"Dataset: {sum(counts):,} rows, 31 columns, {counts[0]:,} legitimate, "
          f"{counts[1]:,} fraud ({counts[1] / sum(counts) * 100:.4f}%)", flush=True)
    print(f"Features: 30 | Device: {device} | Train: {len(train):,} | Test: {len(test):,}\n"
          f"Contamination prior: {CONTAMINATION:.6f}", flush=True)
    modes = [('primary_unsupervised', train)]
    if args.compare_normal_only:
        modes.append(('normal_only_reference', train[y_train.to_numpy() == 0]))
    for name, training in modes:
        print(f'\nTraining {name}...', flush=True)
        model = build_model(device, args.epochs, args.batch_size)
        t0 = time.perf_counter()
        model.fit(training)  # No label is passed to either model.
        duration = time.perf_counter() - t0
        scores = model.decision_function(test)
        predictions = (scores > model.threshold_).astype(int)
        if not np.isfinite(scores).all() or np.ptp(scores) == 0:
            raise ValueError('Model returned non-finite or constant anomaly scores')
        metrics = evaluate(y_test, scores, predictions)
        metrics.update({'threshold': float(model.threshold_), 'training_seconds': duration,
                        'training_rows': len(training)})
        info[name] = metrics
        prefix = '' if name == 'primary_unsupervised' else 'normal_only_'
        save_plots(out, y_test.to_numpy(), scores, metrics, [counts[0], counts[1]], prefix)
        pd.DataFrame({'original_row_index': x_test.index, 'anomaly_score': scores,
                      'predicted_anomaly': predictions, 'actual_Class': y_test.to_numpy()}
                     ).nlargest(50, 'anomaly_score').to_csv(out / f'{prefix}top_anomalies.csv', index=False)
        print(f"Training complete ({duration:.1f}s). Threshold: {model.threshold_:.6f}\n"
              f"ROC-AUC: {metrics['roc_auc']:.4f} | Average precision: {metrics['average_precision']:.4f}\n"
              f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}\n"
              f"TP: {metrics['tp']} | FP: {metrics['fp']} | TN: {metrics['tn']} | FN: {metrics['fn']}\n"
              f"Predicted anomalies: {metrics['predicted_anomalies']}", flush=True)
    info['total_seconds'] = time.perf_counter() - start
    summary = json.dumps(info, indent=2, allow_nan=False) + '\n'
    (out / 'metrics.json').write_text(summary)
    if args.compare_normal_only:
        (out / 'metrics_comparison.json').write_text(summary)
    primary = info['primary_unsupervised']
    print('\n' + '=' * 60 + '\n PyOD AutoEncoder Fraud Detection | MSCS-633', flush=True)
    if args.quick:
        print(' QUICK / DEVELOPMENT MODE — NOT FINAL RESULTS', flush=True)
    print(f" Dataset: {sum(counts):,} rows | Fraud: {counts[1]:,} ({counts[1] / sum(counts) * 100:.4f}%)\n"
          f" Primary unsupervised | Device: {device} | Train: {len(train):,} | Test: {len(test):,}\n"
          f" Training complete: {primary['training_seconds']:.1f}s | Threshold: {primary['threshold']:.6f}\n"
          f" ROC-AUC: {primary['roc_auc']:.4f} | Average precision: {primary['average_precision']:.4f}\n"
          f" Precision: {primary['precision']:.4f} | Recall: {primary['recall']:.4f} | F1: {primary['f1']:.4f}\n"
          f" TP: {primary['tp']} | FP: {primary['fp']} | TN: {primary['tn']} | FN: {primary['fn']}\n"
          f" Artifacts: {out.relative_to(ROOT)}/ (metrics.json, plots, top_anomalies.csv)\n"
          ' Experiment complete.\n' + '=' * 60, flush=True)


def main():
    """Parse arguments and report actionable failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATASET_PATH)
    parser.add_argument('--device', choices=['auto', 'mps', 'cpu'], default='auto')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--compare-normal-only', action='store_true')
    parser.add_argument('--epochs', type=positive_int, default=EPOCHS)
    parser.add_argument('--batch-size', type=positive_int, default=BATCH_SIZE)
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'Experiment failed: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
