from extract_features import Classification_FeatureExtractor
from classification_process import MLPClassifierPipeline

"""
1. Extract the eigenvectors of the amino acid sequence
2. Select the appropriate model parameters through cross-validation and mesh search
3. The performance of the model trained at each fold is evaluated
4. Save the best classification model
5. Plot the confusion matrix on the validation set and the independent test set
6. UMAP dimensionality reduction visualization
"""


# Feature extract
classification_extractor = Classification_FeatureExtractor(
    model_file='checkpoints/esm2_t33_650M_UR50D.pt',
    regression_weights_file='checkpoints/esm2_t33_650M_UR50D-contact-regression.pt'
)
classification_extractor.extract(
    data_file='data-classification/train.fasta',
    output_file='output-classification/train.csv'
)
classification_extractor.extract(
    data_file='data-classification/test.fasta',
    output_file='output-classification/test.csv'
)

# Model training
pipeline = MLPClassifierPipeline()

# Load the training/test set feature data
X_train, y_train = pipeline.load_dataset('output-classification/train.csv')
X_test, y_test = pipeline.load_dataset('output-classification/test.csv')

# Tune parameter & Training
pipeline.tune_and_validate(X_train, y_train)

# Evaluate
pipeline.evaluate_on_per_fold(X_train, y_train, X_test, y_test)

# Save the model
pipeline.save_model('best_model.pkl')

# Confusion matrix
pipeline.confusion_matrix_per_fold(X_train, y_train, X_test, y_test, class_names=["Negative", "Positive"])

# UMAP dimensionality reduction and visualization of the original features
pipeline.plot_umap(X_test, y_test)

# The hidden layer output of the model was extracted and visualized by UMAP dimensionality reduction
pipeline.plot_umap_after_model(X_test, y_test, model_path='best_model.pkl')
