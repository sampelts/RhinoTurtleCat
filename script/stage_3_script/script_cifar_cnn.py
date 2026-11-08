
import sys
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / 'data' / 'stage_3_data'
RESULT_DIR = PROJECT_ROOT / 'result' / 'stage_3_result'

from local_code.stage_3_code.Dataset_Loader import Dataset_Loader_CIFAR
from local_code.stage_3_code.Method_CNN_CIFAR import Method_CNN_CIFAR
from local_code.stage_3_code.Setting_Train_Test_PreSplit import Setting_Train_Test_PreSplit
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.History_Saver import History_Saver


def main():
    np.random.seed(2)
    torch.manual_seed(2)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # Training data
    train_obj = Dataset_Loader_CIFAR(
        split='train',
        data_dir=DATA_DIR
    )

    # Testing data
    test_obj = Dataset_Loader_CIFAR(
        split='test',
        data_dir=DATA_DIR
    )

    # CNN model
    method_obj = Method_CNN_CIFAR(
        'CNN CIFAR',
        'CIFAR-10 image classification'
    )

    # Save results
    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = str(RESULT_DIR)
    result_obj.result_destination_file_name = 'CIFAR_predictions'
    result_obj.fold_count = 0

    # Evaluation
    evaluate_obj = Evaluate_Accuracy(num_classes=10, label_start=0)

    # Experiment setting
    setting_obj = Setting_Train_Test_PreSplit(
        'Presplit dataset',
        'CIFAR-10'
    )

    setting_obj.prepare(
        train_obj,
        test_obj,
        method_obj,
        evaluate_obj,
        result_obj
    )

    print("******** CIFAR-10 CNN ********")

    results = setting_obj.load_run_save_evaluate()

    print("Accuracy:", results['accuracy'])
    print("Macro Precision:", results['precision_macro'])
    print("Macro Recall:", results['recall_macro'])
    print("Macro F1:", results['f1_macro'])

    # Save training history for learning curves
    history_obj = History_Saver()
    history_obj.history_destination_folder_path = str(RESULT_DIR)
    history_obj.history_destination_file_name = 'CIFAR_history.csv'
    history_obj.data = method_obj.history
    history_obj.save()


if __name__ == '__main__':
    main()
