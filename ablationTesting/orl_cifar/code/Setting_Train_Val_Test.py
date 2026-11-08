'''
Concrete SettingModule: carve a stratified validation set out of the supplied
training split, train on the rest, compare configurations on validation, and
evaluate the untouched supplied test split exactly once at the end.
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting
from sklearn.model_selection import train_test_split
import numpy as np


class Setting_Train_Val_Test(setting):
    val_fraction = 0.1
    split_seed = 0

    def split_train(self, X, y):
        n_classes = len(np.unique(y))
        # at least one validation image per class (ORL has only 9 training images per person)
        n_val = max(int(round(self.val_fraction * len(y))), n_classes)
        idx_train, idx_val = train_test_split(
            np.arange(len(y)), test_size=n_val, stratify=y, random_state=self.split_seed)
        return {'X': X[idx_train], 'y': y[idx_train]}, {'X': X[idx_val], 'y': y[idx_val]}

    def load_run_save_evaluate(self):
        # load dataset (already split into train/test by the instructor)
        data = self.dataset.load()
        train, val = self.split_train(data['train']['X'], data['train']['y'])
        test = data['test']

        # run MethodModule: Method_CNN draws its per-epoch curves on the 'test' entry,
        # so we hand it the validation split; the real test split stays untouched
        self.method.data = {'train': train, 'test': val}
        val_result = self.method.run()
        self.evaluate.data = val_result
        val_metrics = self.evaluate.evaluate()

        # final, single evaluation on the supplied test split
        test_result = {'pred_y': self.method.test(test['X']), 'true_y': test['y']}
        self.evaluate.data = test_result
        test_metrics = self.evaluate.evaluate()

        # save raw ResultModule (test predictions)
        self.result.data = test_result
        self.result.save()

        return {'val': val_metrics, 'test': test_metrics,
                'n_train': len(train['y']), 'n_val': len(val['y']), 'n_test': len(test['y'])}
