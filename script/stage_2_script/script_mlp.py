from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / 'data' / 'stage_2_data'
RESULT_DIR = PROJECT_ROOT / 'sigmoid activation results'

from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
from local_code.stage_2_code.Setting_Train_Test_PreSplit import Setting_Train_Test_PreSplit
from local_code.stage_2_code.History_Saver import History_Saver
import numpy as np
import torch

#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    data_train_obj = Dataset_Loader('train', '')
    data_train_obj.dataset_source_folder_path = str(DATA_DIR) + '/'
    data_train_obj.dataset_source_file_name = 'train.csv'

    data_test_obj = Dataset_Loader('test', '')
    data_test_obj.dataset_source_folder_path = str(DATA_DIR) + '/'
    data_test_obj.dataset_source_file_name = 'test.csv'

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    history_obj = History_Saver('', '')
    history_obj.history_destination_folder_path = str(RESULT_DIR) + '/'
    history_obj.history_destination_file_name = 'history_MLP.csv'

    method_obj = Method_MLP('multi-layer perceptron', '')
    method_obj.history = history_obj

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = str(RESULT_DIR / 'MLP_')
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test_PreSplit('Presplit dataset', 'mnist presplit dataset')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')

    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_train_obj, data_test_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('MLP Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('************ Finish ************')
    # ------------------------------------------------------
