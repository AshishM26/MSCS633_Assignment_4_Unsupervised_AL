"""Plot held-out results with labels appropriate for imbalanced data."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay


def save_plots(out, y, scores, metrics, counts, prefix=''):
    """Write class, score, confusion matrix, ROC, and precision-recall plots."""
    def save(name):
        plt.tight_layout()
        plt.savefig(out / f'{prefix}{name}.png', dpi=150)
        plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(['Legitimate', 'Fraud'], counts, color=['#377eb8', '#e41a1c'])
    ax.set_yscale('log'); ax.set_ylabel('Transactions (log scale)')
    ax.set_title('Full dataset class distribution')
    for i, v in enumerate(counts):
        ax.text(i, v * 1.15, f'{v:,}', ha='center')
    ax.set_ylim(top=max(counts) * 5)
    save('class_distribution')
    fig, ax = plt.subplots(figsize=(7, 4))
    # log1p includes zero and compresses the long positive score tail.
    for label, color in [(0, '#377eb8'), (1, '#e41a1c')]:
        ax.hist(np.log1p(scores[np.asarray(y) == label]), bins=80, density=True,
                alpha=0.55, label=['Legitimate', 'Fraud'][label], color=color)
    ax.axvline(np.log1p(metrics['threshold']), color='black', linestyle='--', label='Threshold')
    ax.set(xlabel='log(1 + anomaly score)', ylabel='Density', title='Held-out anomaly scores')
    ax.legend(); save('anomaly_score_distribution')
    fig, ax = plt.subplots(figsize=(5, 4))
    matrix = np.array([[metrics['tn'], metrics['fp']], [metrics['fn'], metrics['tp']]])
    ax.imshow(matrix, cmap='Blues')
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f'{value:,}', ha='center', va='center',
                color='white' if value > matrix.max() / 2 else 'black')
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Normal', 'Anomaly'],
           yticklabels=['Legitimate', 'Fraud'], xlabel='Predicted', ylabel='Actual',
           title='Held-out confusion matrix')
    save('confusion_matrix')
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y, scores, ax=ax)
    ax.plot([0, 1], [0, 1], '--', color='gray', label='Random')
    ax.set_title('Held-out ROC curve'); ax.legend(); save('roc_curve')
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y, scores, ax=ax)
    ax.axhline(np.mean(y), linestyle='--', color='gray', label='Fraud prevalence')
    ax.set_title('Held-out precision-recall curve'); ax.legend(); save('precision_recall_curve')
