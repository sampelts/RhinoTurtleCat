'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting
from sklearn.model_selection import train_test_split
import numpy as np

class Setting_Train_Test_PreSplit(setting):
    fold = 3

    def prepare(self, sTrainSet, sTestSet, sMethod, sResult, sEvaluate):
        self.trainSet = sTrainSet
        self.testSet = sTestSet
        # assigned for compatibility
        self.dataset = sTrainSet
        self.method = sMethod
        self.result = sResult
        self.evaluate = sEvaluate

    def load_run_save_evaluate(self):
        
        # load dataset
        loaded_data_train = self.trainSet.load()
        loaded_data_test = self.testSet.load()

        X_train, y_train = loaded_data_train['X'], loaded_data_train['y']
        X_test, y_test = loaded_data_test['X'], loaded_data_test['y']

        # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        learned_result = self.method.run()
            
        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()
            
        self.evaluate.data = learned_result
        
        return self.evaluate.evaluate(), 0.0

