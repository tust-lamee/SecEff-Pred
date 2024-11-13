from fasta_to_dataframe import Fasta_to_Dataframe
from autogluon.multimodal import MultiModalPredictor
import warnings
import os
import shutil
import numpy as np

warnings.filterwarnings("ignore")
np.random.seed(123)

class Dataset(Fasta_to_Dataframe):
    pass

train_data = Dataset.fasta_to_dataframe('train_4.fasta')
test_data = Dataset.fasta_to_dataframe('test.fasta')
val_data = Dataset.fasta_to_dataframe('val_4.fasta')

def SP_Predict():
    label = 'Label'
    save_path = 'classification_model'

    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    predictor = MultiModalPredictor(label=label, eval_metric='roc_auc', path=save_path)
    predictor.fit(train_data, tuning_data=val_data, hyperparameters={
        "model.hf_text.checkpoint_name": "model_checkpoint",
                  },)

    test_result = predictor.evaluate(test_data, metrics=['acc', 'recall', 'precision', 'roc_auc', 'mcc'])
    print(test_result)
    val_result = predictor.evaluate(val_data, metrics=['acc', 'recall', 'precision', 'roc_auc', 'mcc'])
    print(val_result)


if __name__ == '__main__':
    SP_Predict()

