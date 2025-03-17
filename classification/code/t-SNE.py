from fasta_to_dataframe import Fasta_to_Dataframe
from sklearn.manifold import TSNE
from autogluon.multimodal import MultiModalPredictor
import matplotlib.pyplot as plt
import warnings
from matplotlib import rcParams

warnings.filterwarnings("ignore")

class Dataset(Fasta_to_Dataframe):
    pass

test_data = Dataset.fasta_to_dataframe('test.fasta')

predictor = MultiModalPredictor.load('./classification_model')
embeddings = predictor.extract_embedding(test_data)

tsne = TSNE(n_components=2, random_state=22)
X_tsne = tsne.fit_transform(embeddings)

save_path = 't-SNE.png'
target_id = ['0', '1']
colors = ['r', 'g']
fig, ax = plt.subplots()
for i, c, label in zip(target_id, colors, ['bad-performing', 'good-performing']):
    idx = (test_data['Label'].to_numpy() == i).nonzero()
    plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], s=12, alpha=0.6, c=c, label=label)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(loc='upper right')
    plt.savefig(save_path, dpi=600)
