from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / 'data' / 'stage_3_data'
RESULT_DIR = PROJECT_ROOT / 'result' / 'stage_3_result'

from local_code.stage_3_code.Dataset_Loader_Image import Dataset_Loader
from local_code.stage_3_code.Method_CNN_ORL import Method_CNN_ORL
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
from local_code.stage_3_code.Setting_Train_Test_PreSplit import Setting_Train_Test_PreSplit
from local_code.stage_3_code.History_Saver import History_Saver
import numpy as np
import torch

#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    torch.set_num_threads(16)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    dataset_obj = Dataset_Loader('orl', 'ORL Dataset')
    dataset_obj.dataset_source_folder_path = '../../data/stage_3_data'
    dataset_obj.dataset_source_file_name = 'ORL'
    dataset_obj.channel_index = 0   # 0-R channel, 1-G, 2-B
    dataset_obj.label_offset = 1    # ORL labels shifted by 1

    if False:
        train_ds, test_ds = dataset_obj.load()
        x, y = train_ds.tensors
        print(len(train_ds), len(test_ds))  # 360 40
        print(x.shape, x.dtype, y.dtype)  # [360, 1, 112, 92] float32 int64
        print(x.mean().item(), x.std().item())  # ~0.0, ~1.0
        print(y.min().item(), y.max().item())  # 0 39
        print(test_ds.tensors[0].mean().item())  # close to 0, not exactly 0
        exit(0)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    history_obj = History_Saver('', '')
    history_obj.history_destination_folder_path = '../../result/stage_3_result/'
    history_obj.history_destination_file_name = 'history_orl.csv'

    method_obj = Method_CNN_ORL('CNN', 'ORL model')
    method_obj.history = history_obj

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_3_result/CNN_ORL'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test_PreSplit('orl dataset', '')

    evaluate_obj = Evaluate_Accuracy('All metrics', '')



    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(dataset_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('CNN_ORL Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('************ Finish ************')
    # ------------------------------------------------------
