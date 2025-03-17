from autogluon.multimodal import MultiModalPredictor
import matplotlib.pyplot as plt
from fasta_to_dataframe import Fasta_to_Dataframe
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
from matplotlib import rcParams

class Dataset(Fasta_to_Dataframe):
    pass

test_data = Dataset.fasta_to_dataframe('test.fasta')

predictor = MultiModalPredictor.load('./classification_model')

y_true = list(map(int, list(test_data['Label'])[:]))
y_pred = list(map(int, np.array(predictor.predict(test_data))))

C = confusion_matrix(y_true, y_pred)

save_path = 'confusion_matrix_test.png'

fig, ax = plt.subplots()

sns.heatmap(C, cmap='Blues', annot=True, fmt='d')

ax.set(yticklabels = ['Negative', 'Positive'],
       xticklabels = ['Negative', 'Positive'])

plt.ylabel('True label', fontdict={'size': 15})
plt.xlabel('Predicted label', fontdict={'size': 15})
plt.title('Confusion matrix', fontdict={'size': 20})
plt.savefig(save_path, dpi=600)
