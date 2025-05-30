SecEff-Pred is a novel web server using transformer architecture to predict the secretion efficiency of signal peptides in Bacillus subtilis, filling a significant gap in the field. The SecEff-Pred web service is publicly accessible at http://www.lamee.cn/web_service/.

1. extract_features.py 
Sequence features are extracted using ESM pre-trained model. Among them, the input file of the regression task is in csv format, the input file of the classification task is in FASTA format, and the extracted feature vectors are written to the csv format file.
Please use the following download URL for the pretrained model: https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt and place it into the checkpoints folder.

2. classification_process.py
For the classification task, we define the following methods:
(1) load_dataset: Load the feature dataset in CSV format.
(2) tune_and_validate: Use grid search and cross-validation of the tuning parameters MLP model.
(3) evaluate_on_per_fold: Train the model on each fold and evaluate the performance of the validation and test sets.
(4) save_model: Save the best model to the specified file.
(5) predict: Use the trained model to make predictions on FASTA sequences.
(6) confusion_matrix_per_fold: Confusion matrix plots for the validation set and the test set are generated separately on each fold.
(7) plot_umap: Normalize and UMAP dimensionality reduction and visualization of original features.
(8) plot_umap_after_model: Extract the hidden layer output of the model and visualize the dimensionality reduction of UMAP.

3. regression.py
Construct a regression model to predict the secretion efficiency value of signal peptides.

4. classification.py
Call the method in classification_process.py to build a classification model to distinguish high secretion efficiency and low secretion efficiency signal peptides.
