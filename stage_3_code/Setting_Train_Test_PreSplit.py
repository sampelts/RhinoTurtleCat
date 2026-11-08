
from local_code.base_class.setting import setting


class Setting_Train_Test_PreSplit(setting):

    def __init__(
        self,
        sName="Train Test PreSplit",
        sDescription="Pre-split training and testing"
    ):
        super().__init__(sName, sDescription)

        self.trainSet = None
        self.testSet = None
        self.method = None
        self.evaluate = None
        self.result = None

    def prepare(
        self,
        train_data,
        test_data,
        method_obj,
        evaluate_obj,
        result_obj
    ):
        # Store training and testing datasets
        self.trainSet = train_data
        self.testSet = test_data

        # Store CNN, evaluator, and result saver
        self.method = method_obj
        self.evaluate = evaluate_obj
        self.result = result_obj

    def load_run_save_evaluate(self):

        # Load the instructor's pre-split datasets
        train_data = self.trainSet.load()
        test_data = self.testSet.load()

        # Pass both datasets into the CNN method
        self.method.data = {
            "train": train_data,
            "test": test_data
        }

        # Train CNN and obtain testing predictions
        results = self.method.run()

        # Evaluate predictions
        self.evaluate.data = results
        metrics = self.evaluate.evaluate()

        print("Evaluation Results:")
        print(metrics)

        # Save predictions and evaluation metrics
        if self.result is not None:
            self.result.data = {
                "predictions": results,
                "metrics": metrics
            }

            self.result.save()

        return metrics
