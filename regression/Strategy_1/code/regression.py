import pandas as pd
from autogluon.multimodal import MultiModalPredictor
import warnings
import os
import shutil
import numpy as np

warnings.filterwarnings("ignore")
np.random.seed(123)

test_data = pd.read_csv('test.csv')
train_data = pd.read_csv('train_8.csv')
val_data = pd.read_csv('val_8.csv')

def Train_Regression():
    label = 'Label'
    save_path = 'regression_model'

    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    predictor = MultiModalPredictor(label=label, eval_metric='root_mean_squared_error',
                                    path=save_path, problem_type='regression')
    predictor.fit(train_data, tuning_data=val_data, hyperparameters={
                      "model.hf_text.checkpoint_name": "model_checkpoint",
                  },)

    test_result = predictor.evaluate(test_data, metrics=['r2', 'pearsonr', 'rmse'])
    print(test_result)
    val_result = predictor.evaluate(val_data, metrics=['r2', 'pearsonr', 'rmse'])
    print(val_result)


if __name__ == '__main__':
    Train_Regression()
