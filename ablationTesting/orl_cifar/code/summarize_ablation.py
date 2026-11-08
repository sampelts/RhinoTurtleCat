'''
Turn result/stage_3_ablation/summary.csv into report-ready tables and figures.

Outputs (written next to summary.csv):
    ablation_table.md / .csv   one row per configuration, one column per dataset (section 3.7)
    baseline_metrics.md / .csv accuracy, precision, recall, F1 of each baseline (section 3.6)
    convergence.png            baseline train/validation loss per epoch, one panel per dataset (section 3.5)

usage:
    python script/stage_3_script/summarize_ablation.py
    python script/stage_3_script/summarize_ablation.py --out-dir /content/drive/MyDrive/stage3_ablation
'''

from pathlib import Path
import argparse
import csv
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ORDER = ['MNIST', 'ORL', 'CIFAR']
CONFIG_ORDER = ['baseline', 'shallower', 'deeper', 'kernel_5', 'no_padding', 'stride_2', 'avg_pool',
                'hidden_64', 'batch_norm', 'no_dropout', 'margin_loss', 'sgd']


def fmt(mean, std, n):
    '''Percent with one decimal; add +/- std when several seeds were run.'''
    if pd.isna(mean):
        return '-'
    return f'{100 * mean:.1f} ± {100 * std:.1f}' if n > 1 else f'{100 * mean:.1f}'


def to_markdown(df):
    lines = ['| ' + ' | '.join(df.columns) + ' |', '|' + '---|' * len(df.columns)]
    lines += ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in df.itertuples(index=False)]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out-dir', type=Path, default=PROJECT_ROOT / 'result' / 'stage_3_ablation')
    args = parser.parse_args()
    summary_path = args.out_dir / 'summary.csv'
    if not summary_path.exists():
        sys.exit(f'{summary_path} not found; run script_cnn_ablation.py first')

    runs = pd.read_csv(summary_path)
    datasets = [d for d in DATASET_ORDER if d in set(runs['dataset'])]
    stats = (runs.groupby(['dataset', 'config'])
             .agg(test_mean=('test_accuracy', 'mean'), test_std=('test_accuracy', 'std'),
                  val_mean=('val_accuracy', 'mean'), val_std=('val_accuracy', 'std'),
                  seeds=('seed', 'count'), change=('change', 'first'))
             .fillna({'test_std': 0.0, 'val_std': 0.0}))

    # ---- section 3.7: one row per configuration, one column per dataset -------------
    configs = [c for c in CONFIG_ORDER if c in set(runs['config'])]
    rows, csv_rows = [], []
    for config in configs:
        change = runs.loc[runs['config'] == config, 'change'].iloc[0]
        row, csv_row = {'Configuration': change}, {'config': config, 'change': change}
        for d in datasets:
            if (d, config) in stats.index:
                s = stats.loc[(d, config)]
                cell = fmt(s.test_mean, s.test_std, s.seeds)
                if config != 'baseline' and (d, 'baseline') in stats.index:
                    delta = 100 * (s.test_mean - stats.loc[(d, 'baseline')].test_mean)
                    cell += f' ({delta:+.1f})'
                row[f'{d} test acc %'] = cell
                row[f'{d} val acc %'] = fmt(s.val_mean, s.val_std, s.seeds)
                csv_row.update({f'{d}_test_acc': s.test_mean, f'{d}_test_std': s.test_std,
                                f'{d}_val_acc': s.val_mean, f'{d}_val_std': s.val_std, f'{d}_seeds': s.seeds})
            else:
                row[f'{d} test acc %'] = row[f'{d} val acc %'] = '-'
        rows.append(row)
        csv_rows.append(csv_row)
    table = pd.DataFrame(rows)
    # compact version for the report: test accuracy only
    compact = table[['Configuration'] + [f'{d} test acc %' for d in datasets]]
    note = ('\nTest accuracy in %, difference from the baseline in parentheses; '
            '"± x" is the standard deviation over seeds. Configurations were compared on the '
            'validation split (full table below); the supplied test split was evaluated once per run.\n')
    (args.out_dir / 'ablation_table.md').write_text(
        '## Configuration experiments (compact, for the report)\n\n' + to_markdown(compact) + note +
        '\n## Full table (test and validation accuracy)\n\n' + to_markdown(table))
    pd.DataFrame(csv_rows).to_csv(args.out_dir / 'ablation_table.csv', index=False)

    # ---- section 3.6: baseline accuracy / precision / recall / F1 -----------------------
    base = runs[runs['config'] == 'baseline']
    metric_cols = ['test_accuracy', 'test_precision_macro', 'test_recall_macro', 'test_f1_macro']
    perf_rows = []
    for d in datasets:
        part = base[base['dataset'] == d]
        if part.empty:
            continue
        n = len(part)
        perf_rows.append({'Dataset': d, 'Seeds': n,
                          **{c.replace('test_', '').replace('_macro', ' (macro)').capitalize() + ' %':
                             fmt(part[c].mean(), part[c].std() if n > 1 else 0.0, n) for c in metric_cols}})
    perf = pd.DataFrame(perf_rows)
    (args.out_dir / 'baseline_metrics.md').write_text(
        '## Baseline model performance on the supplied test split\n\n' + to_markdown(perf))
    perf.to_csv(args.out_dir / 'baseline_metrics.csv', index=False)

    # ---- section 3.5: convergence plot, x = epoch, y = loss -----------------------------
    fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 3.2), squeeze=False)
    for ax, d in zip(axes[0], datasets):
        seed_dirs = sorted((args.out_dir / d / 'baseline').glob('seed*/history.csv'))
        if not seed_dirs:
            ax.set_visible(False)
            continue
        with open(seed_dirs[0]) as f:
            hist = list(csv.DictReader(f))
        epochs = [int(r['Epoch']) + 1 for r in hist]
        ax.plot(epochs, [float(r['Train_Loss']) for r in hist], label='train loss')
        ax.plot(epochs, [float(r['Val_Loss']) for r in hist], '--', label='validation loss')
        ax.set(title=f'{d} CNN (baseline)', xlabel='Training epoch', ylabel='Cross-entropy loss')
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / 'convergence.png', dpi=200)
    plt.close(fig)

    print((args.out_dir / 'ablation_table.md').read_text())
    print((args.out_dir / 'baseline_metrics.md').read_text())
    print('figures and tables written to', args.out_dir)


if __name__ == '__main__':
    main()
