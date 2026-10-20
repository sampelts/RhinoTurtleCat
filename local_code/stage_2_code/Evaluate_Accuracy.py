'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class Evaluate_Accuracy(evaluate):
    data = None
    
    def evaluate(self):
        true_y = self.data["true_y"]
        pred_y = self.data["pred_y"]

        precision, recall, f1, _ = precision_recall_fscore_support(
            true_y,
            pred_y,
            labels=list(range(10)),
            average="macro",
            zero_division=0,
        )

        return {
            "accuracy": accuracy_score(true_y, pred_y),
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
        }
        