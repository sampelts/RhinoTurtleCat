'''
Stage 3 CNN script for the ORL (faces) dataset.

usage:
    python script/stage_3_script/script_cnn_orl.py
    python script/stage_3_script/script_cnn_orl.py --config deeper
    python script/stage_3_script/script_cnn_orl.py --config all
'''

from pathlib import Path
import argparse
import csv
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / 'data' / 'stage_3_data'
RESULT_DIR = PROJECT_ROOT / 'result' / 'stage_3_result'

from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN_ORL import Method_CNN_ORL
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
from local_code.stage_3_code.Setting_Train_Test_PreSplit import Setting_Train_Test_PreSplit
from local_code.stage_3_code.History_Saver import History_Saver
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

DATASET_NAME = 'ORL'

# configuration experiments (step 3-5): each one changes a single setting of the baseline
CONFIGS = {
    'baseline': {},
    'shallower': {'conv_channels': (16, 32)},
    'deeper': {'conv_channels': (16, 32, 64, 128)},
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


def plot_learning_curves(rows, path, title):
    epochs = [r['Epoch'] + 1 for r in rows]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    ax_loss.plot(epochs, [r['Train_Loss'] for r in rows], label='train')
    ax_loss.plot(epochs, [r['Test_Loss'] for r in rows], label='test')
    ax_loss.set_xlabel('epoch')
    ax_loss.set_ylabel('loss')
    ax_loss.set_title('Loss')
    ax_loss.legend()
    ax_acc.plot(epochs, [r['Train_Accuracy'] for r in rows], label='train')
    ax_acc.plot(epochs, [r['Test_Accuracy'] for r in rows], label='test')
    ax_acc.set_xlabel('epoch')
    ax_acc.set_ylabel('accuracy')
    ax_acc.set_title('Accuracy')
    ax_acc.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_experiment(config_name, epochs):
    dataset_name = DATASET_NAME
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    run_dir = RESULT_DIR / (dataset_name + '_' + config_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    data_obj = Dataset_Loader(dataset_name, '')
    data_obj.dataset_source_folder_path = str(DATA_DIR) + '/'
    data_obj.dataset_source_file_name = dataset_name
    data_obj.grayscale = True

    history_obj = History_Saver('', '')
    history_obj.history_destination_folder_path = str(run_dir) + '/'
    history_obj.history_destination_file_name = 'history_CNN.csv'

    overrides = dict(CONFIGS[config_name])
    if epochs is not None:
        overrides['max_epoch'] = epochs
    method_obj = Method_CNN_ORL('CNN ' + dataset_name, config_name, **overrides)
    method_obj.history = history_obj
    print(method_obj)

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = str(run_dir / 'CNN_')
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test_PreSplit('Presplit dataset', dataset_name + ' presplit dataset')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ' + dataset_name + ' / ' + config_name + ' ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    metrics, _ = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    for k, v in metrics.items():
        print(f'{k}: {v:.4f}')

    plot_learning_curves(history_obj.data, run_dir / 'learning_curve.png',
                         'CNN on ' + dataset_name + ' (' + config_name + ')')
    with open(run_dir / 'metrics.json', 'w') as f:
        json.dump({'config': {k: list(v) if isinstance(v, tuple) else v for k, v in method_obj.config.items()},
                   'test_metrics': metrics}, f, indent=2)

    # append one row per run so the configurations can be compared in the report
    summary_path = RESULT_DIR / 'summary.csv'
    new_file = not summary_path.exists()
    with open(summary_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(['dataset', 'config'] + list(metrics))
        writer.writerow([dataset_name, config_name] + [f'{v:.4f}' for v in metrics.values()])
    print('************ Finish ************')
    # ------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', choices=sorted(CONFIGS) + ['all'], default='baseline')
    parser.add_argument('--epochs', type=int, default=None, help='override the number of epochs')
    args = parser.parse_args()

    config_names = list(CONFIGS) if args.config == 'all' else [args.config]
    for name in config_names:
        run_experiment(name, args.epochs)
