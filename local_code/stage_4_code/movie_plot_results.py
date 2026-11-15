"""Create report figures and comparison tables from completed experiments."""
import csv
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ecs170-stage4-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_results(folder):
    folder = Path(folder)
    metrics = json.loads((folder / 'comparison.json').read_text())
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 3.3), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        with (folder / metric['cell'].lower() / 'history.csv').open() as f:
            rows = list(csv.DictReader(f))
        epochs = [int(row['epoch']) for row in rows]
        ax.plot(epochs, [float(row['train_loss']) for row in rows], label='Training', marker='o', markersize=3)
        ax.plot(epochs, [float(row['val_loss']) for row in rows], label='Validation', marker='o', markersize=3)
        ax.axvline(metric['best_epoch'], color='gray', linestyle=':', linewidth=1, label='Selected epoch')
        ax.set(title=metric['cell'], xlabel='Epoch', ylabel='Binary cross-entropy')
        ax.set_xticks(epochs)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / 'learning_curves.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.4), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        matrix = np.asarray(metric['confusion_matrix'])
        ax.imshow(matrix, cmap='Blues', vmin=0)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i, j]), ha='center', va='center',
                        color='white' if matrix[i, j] > matrix.max() / 2 else 'black')
        ax.set(title=metric['cell'], xlabel='Predicted label', ylabel='True label',
               xticks=[0, 1], yticks=[0, 1], xticklabels=['Negative', 'Positive'],
               yticklabels=['Negative', 'Positive'])
    fig.tight_layout()
    fig.savefig(folder / 'confusion_matrices.png', dpi=180)
    plt.close(fig)
    keys = ('cell', 'accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
            'best_epoch', 'parameter_count', 'training_seconds')
    rows = [{key: m[key] for key in keys} for m in metrics]
    with (folder / 'comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Movie model comparison', '',
             'Smoke test only. These are not final experimental results.' if any(m['smoke_run'] for m in metrics)
             else 'Full held-out test results. Precision, recall and F1 are macro-averaged.', '',
             '| Model | Accuracy | Precision | Recall | F1 | Selected epoch | Parameters |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for m in metrics:
        lines.append(f"| {m['cell']} | {m['accuracy']:.4f} | {m['precision_macro']:.4f} | "
                     f"{m['recall_macro']:.4f} | {m['f1_macro']:.4f} | {m['best_epoch']} | {m['parameter_count']:,} |")
    lines += ['', '![Training and validation loss](learning_curves.png)', '',
              '![Test confusion matrices](confusion_matrices.png)', '']
    (folder / 'comparison.md').write_text('\n'.join(lines))
