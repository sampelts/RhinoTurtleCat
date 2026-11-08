'''
Stage 3 configuration experiments (step 3-5) for MNIST, ORL and CIFAR in one runner.

Each configuration changes exactly one setting of the dataset's baseline CNN.
Models are trained on the supplied training split minus a stratified validation
split, compared on validation, and evaluated once on the untouched test split.

Runs are resumable: a run whose metrics.json already exists is skipped, so a
disconnected Colab session can simply re-run the same command.

usage:
    python script/stage_3_script/script_cnn_ablation.py --dataset MNIST
    python script/stage_3_script/script_cnn_ablation.py --dataset ORL --seeds 0 1 2
    python script/stage_3_script/script_cnn_ablation.py --dataset CIFAR --configs baseline deeper
'''

from pathlib import Path
import argparse
import csv
import json
import random
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from local_code.base_class.dataset import dataset
from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN_MNIST_Ablation import Method_CNN_MNIST_Ablation
from local_code.stage_3_code.Method_CNN_ORL import Method_CNN_ORL
from local_code.stage_3_code.Method_CNN_CIFAR import Method_CNN_CIFAR
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
from local_code.stage_3_code.Setting_Train_Val_Test import Setting_Train_Val_Test
from local_code.stage_3_code.History_Saver import History_Saver
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

# dataset name -> (method class, keep a single grayscale channel)
DATASETS = {
    'MNIST': (Method_CNN_MNIST_Ablation, False),
    'ORL': (Method_CNN_ORL, True),
    'CIFAR': (Method_CNN_CIFAR, False),
}

# configuration name -> what it changes (used in the report table)
CONFIG_DESCRIPTIONS = {
    'baseline': 'Baseline',
    'shallower': 'Depth: one fewer conv block',
    'deeper': 'Depth: one more conv block',
    'kernel_5': 'Kernel size 5x5 (padding 2)',
    'no_padding': 'Padding 0',
    'stride_2': 'Stride 2 conv instead of pooling',
    'avg_pool': 'Average pooling',
    'hidden_64': 'Hidden layer 64 units',
    'batch_norm': 'Add batch normalization',
    'no_dropout': 'No dropout',
    'margin_loss': 'Loss: multi-class margin',
    'sgd': 'Optimizer: SGD (lr 0.01, momentum 0.9)',
}


def make_configs(method_class):
    '''One-change-at-a-time overrides relative to the dataset's own baseline.'''
    base_channels = tuple(method_class.config['conv_channels'])
    return {
        'baseline': {},
        'shallower': {'conv_channels': base_channels[:-1]},
        'deeper': {'conv_channels': base_channels + (base_channels[-1] * 2,)},
        'kernel_5': {'kernel_size': 5, 'padding': 2},
        'no_padding': {'padding': 0},
        'stride_2': {'stride': 2, 'pool': None},
        'avg_pool': {'pool': 'avg'},
        'hidden_64': {'hidden_dim': 64},
        'batch_norm': {'batch_norm': True},
        'no_dropout': {'dropout': 0.0},
        'margin_loss': {'loss': 'multi_margin'},
        'sgd': {'optimizer': 'sgd', 'learning_rate': 0.01},
    }


class Cached_Dataset(dataset):
    '''Load the pickle once and hand the same arrays to every configuration.'''

    def __init__(self, loader):
        # the template's base class spells the attribute 'dataset_descrition'
        super().__init__(loader.dataset_name, getattr(loader, 'dataset_descrition', ''))
        self.loader = loader
        self.cache = None

    def load(self):
        if self.cache is None:
            self.cache = self.loader.load()
        return self.cache


class Val_History_Saver(History_Saver):
    '''Method_CNN labels its held-out curve "Test_*"; here that split is validation.'''

    def save(self):
        self.data = [{k.replace('Test_', 'Val_'): v for k, v in row.items()} for row in self.data]
        super().save()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_curves(rows, path, title):
    epochs = [r['Epoch'] + 1 for r in rows]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 3.6))
    ax_loss.plot(epochs, [r['Train_Loss'] for r in rows], label='train')
    ax_loss.plot(epochs, [r['Val_Loss'] for r in rows], label='validation')
    ax_loss.set(xlabel='epoch', ylabel='loss', title='Loss')
    ax_acc.plot(epochs, [r['Train_Accuracy'] for r in rows], label='train')
    ax_acc.plot(epochs, [r['Val_Accuracy'] for r in rows], label='validation')
    ax_acc.set(xlabel='epoch', ylabel='accuracy', title='Accuracy')
    for ax in (ax_loss, ax_acc):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_summary(out_dir):
    '''Rebuild summary.csv from every finished run, so re-runs never duplicate rows.'''
    rows = [json.loads(p.read_text())['summary'] for p in sorted(out_dir.glob('*/*/seed*/metrics.json'))]
    if not rows:
        return
    with open(out_dir / 'summary.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_one(dataset_name, cached_data, config_name, overrides, seed, args):
    method_class = DATASETS[dataset_name][0]
    run_dir = args.out_dir / dataset_name / config_name / f'seed{seed}'
    if (run_dir / 'metrics.json').exists():
        print(f'skip {dataset_name}/{config_name}/seed{seed} (already done)')
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    overrides = dict(overrides)
    if args.epochs is not None:
        overrides['max_epoch'] = args.epochs

    # ---- object initialization section ------------------
    history_obj = Val_History_Saver('', '')
    history_obj.history_destination_folder_path = str(run_dir)
    history_obj.history_destination_file_name = 'history.csv'

    method_obj = method_class('CNN ' + dataset_name, config_name, **overrides)
    method_obj.history = history_obj

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = str(run_dir) + '/'
    result_obj.result_destination_file_name = 'test_prediction_result'

    setting_obj = Setting_Train_Val_Test('train/val/test', '')
    setting_obj.val_fraction = args.val_fraction
    setting_obj.split_seed = 0  # same validation split for every configuration and seed

    evaluate_obj = Evaluate_Accuracy('accuracy, precision, recall, F1', '')

    # ---- running section ---------------------------------
    print(f'************ {dataset_name} / {config_name} / seed {seed} ************')
    setting_obj.prepare(cached_data, method_obj, result_obj, evaluate_obj)
    started = time.time()
    scores = setting_obj.load_run_save_evaluate()
    seconds = time.time() - started

    rows = history_obj.data
    plot_curves(rows, run_dir / 'learning_curve.png', f'{dataset_name} CNN ({config_name}, seed {seed})')
    config = {k: list(v) if isinstance(v, tuple) else v for k, v in method_obj.config.items()}
    summary = {
        'dataset': dataset_name,
        'config': config_name,
        'change': CONFIG_DESCRIPTIONS[config_name],
        'seed': seed,
        'epochs': config['max_epoch'],
        'parameters': sum(p.numel() for p in method_obj.parameters()),
        'n_train': scores['n_train'], 'n_val': scores['n_val'], 'n_test': scores['n_test'],
        'final_train_loss': rows[-1]['Train_Loss'],
        'final_val_loss': rows[-1]['Val_Loss'],
        'val_accuracy': scores['val']['accuracy'],
        'test_accuracy': scores['test']['accuracy'],
        'test_precision_macro': scores['test']['precision_macro'],
        'test_recall_macro': scores['test']['recall_macro'],
        'test_f1_macro': scores['test']['f1_macro'],
        'test_precision_weighted': scores['test']['precision_weighted'],
        'test_recall_weighted': scores['test']['recall_weighted'],
        'test_f1_weighted': scores['test']['f1_weighted'],
        'seconds': round(seconds, 1),
        'device': str(method_obj.device),
    }
    (run_dir / 'metrics.json').write_text(json.dumps(
        {'summary': summary, 'config': config,
         'val_metrics': {k: float(v) for k, v in scores['val'].items()},
         'test_metrics': {k: float(v) for k, v in scores['test'].items()}}, indent=2))
    print(f"val acc {summary['val_accuracy']:.4f} | test acc {summary['test_accuracy']:.4f} | {seconds:.0f}s")
    write_summary(args.out_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', choices=sorted(DATASETS), required=True)
    parser.add_argument('--configs', nargs='+', default=['all'],
                        help='configuration names, or "all" (default)')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0])
    parser.add_argument('--epochs', type=int, default=None, help='override max_epoch for every configuration')
    parser.add_argument('--val-fraction', type=float, default=0.1)
    parser.add_argument('--data-dir', type=Path, default=PROJECT_ROOT / 'data' / 'stage_3_data')
    parser.add_argument('--out-dir', type=Path, default=PROJECT_ROOT / 'result' / 'stage_3_ablation')
    args = parser.parse_args()

    method_class, grayscale = DATASETS[args.dataset]
    configs = make_configs(method_class)
    names = list(configs) if args.configs == ['all'] else args.configs
    unknown = set(names) - set(configs)
    if unknown:
        parser.error('unknown configs: ' + ', '.join(sorted(unknown)) + '; choose from ' + ', '.join(configs))

    loader = Dataset_Loader(args.dataset, args.dataset + ' presplit dataset')
    loader.dataset_source_folder_path = str(args.data_dir) + '/'
    loader.dataset_source_file_name = args.dataset
    loader.grayscale = grayscale
    cached_data = Cached_Dataset(loader)

    for name in names:
        for seed in args.seeds:
            run_one(args.dataset, cached_data, name, configs[name], seed, args)
    print('************ Finish ************')


if __name__ == '__main__':
    main()
