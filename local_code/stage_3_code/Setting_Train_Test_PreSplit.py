'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting
from sklearn.model_selection import train_test_split
import numpy as np

class Setting_Train_Test_PreSplit(setting):
    dataset = None

    def prepare(self, sDataset, sMethod, sResult, sEvaluate):
        self.dataset = sDataset
        self.method = sMethod
        self.result = sResult
        self.evaluate = sEvaluate

    def load_run_save_evaluate(self):
        
        # load dataset
        loaded_data_train, loaded_data_test = self.dataset.load()

        # run MethodModule
        self.method.data = {'train': loaded_data_train, 'test': loaded_data_test}
        learned_result = self.method.run()
            
        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()
            
        self.evaluate.data = learned_result
        
        return self.evaluate.evaluate(), 0.0

