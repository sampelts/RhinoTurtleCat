'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting


class Setting_Train_Test_PreSplit(setting):
    # the stage 3 dataset files already contain both the train and the test split

    def load_run_save_evaluate(self):

        # load dataset
        self.method.data = self.dataset.load()

        # run MethodModule
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result

        return self.evaluate.evaluate(), 0.0
