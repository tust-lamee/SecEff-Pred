import numpy as np
from sklearn.metrics import confusion_matrix
import umap.umap_ as umap
import seaborn as sns
import matplotlib.pyplot as plt
import random
import csv
import joblib
import warnings
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    matthews_corrcoef, roc_auc_score
)
from sklearn.base import clone


seed = 42
random.seed(seed)
np.random.seed(seed)

class MLPClassifierPipeline:
    def __init__(self):
        """
        Initialize the MLPClassifierPipeline class and define the properties used to store the best model
        """
        self.best_model = None

    def load_dataset(self, path):
        """
        Load the feature dataset in CSV format

        Parameters:
        path: The path to the data file. The first value of each line in the file is a label, followed by a feature

        return:
        vector_x: Feature matrix
        y: Array of tags
        """
        vector_x, y = [], []
        try:
            with open(path, 'r') as file:
                dataset_file = csv.reader(file)
                for content in dataset_file:
                    try:
                        content = list(map(float, content))
                        if len(content) >= 2:
                            y.append(content[0])
                            vector_x.append(content[1:])
                    except ValueError:
                        print(f"Skipping invalid row: {content}")
        except FileNotFoundError:
            print(f"Error: File '{path}' not found.")
            exit(1)
        return np.array(vector_x), np.array(y)

    def tune_and_validate(self, X_train, y_train):
        """
        Use grid search and cross-validation of the tuning parameters MLP model

        Parameters:
        X_train: Training set features
        y_train: Training set labels

        return:
        best_model: Pipeline model with the best combination of parameters
        """
        param_grid = {
            'mlp__hidden_layer_sizes': [
                (128, 64), (256, 128), (512, 256),
                (256, 128, 64), (512, 256, 128)
            ],
            'mlp__alpha': [1e-5, 1e-4, 1e-3, 1e-2],
            'mlp__learning_rate_init': [0.0005, 0.001, 0.005],
            'mlp__activation': ['relu', 'tanh'],
            'mlp__learning_rate': ['constant', 'adaptive']
        }

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(
                solver='adam',
                random_state=1,
                max_iter=2000,
                early_stopping=True,
                validation_fraction=0.2
            ))
        ])

        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv_strategy,
            scoring='accuracy',
            n_jobs=-1,
            refit=True
        )
        grid_search.fit(X_train, y_train)
        self.best_model = grid_search.best_estimator_
        return self.best_model

    def evaluate_on_per_fold(self, X_train, y_train, X_test, y_test):
        """
        Train the model on each fold and evaluate the performance of the validation and test sets

        Parameters:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test : Test labels

        return:
        val_metrics: The evaluation metric for each fold of the validation set
        test_metrics: The evaluation metric for each fold of the independent test set
        """
        if self.best_model is None:
            raise ValueError("Call tune_and_validate() to get the best model first")

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        print("\n[The performance of the training model on the validation and test sets for each fold]")

        test_metrics = {k: [] for k in ['ACC', 'PRE', 'REC', 'MCC', 'AUC']}
        val_metrics = {k: [] for k in ['ACC', 'PRE', 'REC', 'MCC', 'AUC']}

        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_val, y_val = X_train[val_idx], y_train[val_idx]

            model = clone(self.best_model)
            model.fit(X_tr, y_tr)

            # Validation set
            y_val_pred = model.predict(X_val)
            y_val_proba = model.predict_proba(X_val)[:, 1]
            val_metrics['ACC'].append(accuracy_score(y_val, y_val_pred))
            val_metrics['PRE'].append(precision_score(y_val, y_val_pred))
            val_metrics['REC'].append(recall_score(y_val, y_val_pred))
            val_metrics['MCC'].append(matthews_corrcoef(y_val, y_val_pred))
            val_metrics['AUC'].append(roc_auc_score(y_val, y_val_proba))

            # Test set
            y_test_pred = model.predict(X_test)
            y_test_proba = model.predict_proba(X_test)[:, 1]
            test_metrics['ACC'].append(accuracy_score(y_test, y_test_pred))
            test_metrics['PRE'].append(precision_score(y_test, y_test_pred))
            test_metrics['REC'].append(recall_score(y_test, y_test_pred))
            test_metrics['MCC'].append(matthews_corrcoef(y_test, y_test_pred))
            test_metrics['AUC'].append(roc_auc_score(y_test, y_test_proba))

            print(f"Fold {fold}:")
            print(f"  Validation set: ACC={val_metrics['ACC'][-1]:.4f} | PRE={val_metrics['PRE'][-1]:.4f} | "
                  f"REC={val_metrics['REC'][-1]:.4f} | MCC={val_metrics['MCC'][-1]:.4f} | AUC={val_metrics['AUC'][-1]:.4f}")
            print(f"  Test set: ACC={test_metrics['ACC'][-1]:.4f} | PRE={test_metrics['PRE'][-1]:.4f} | "
                  f"REC={test_metrics['REC'][-1]:.4f} | MCC={test_metrics['MCC'][-1]:.4f} | AUC={test_metrics['AUC'][-1]:.4f}")

        return val_metrics, test_metrics

    def save_model(self, filename='best_mlp_model.pkl'):
        """
        Save the best model to the specified file

        Parameters:
        filename: Model save path (default: 'best_mlp_model.pkl')
        """
        if self.best_model is None:
            raise ValueError("If you don't have a model to save, train the model first")
        joblib.dump(self.best_model, filename)
        print(f"The model has been saved to {filename}")

    def predict(self, fasta_path, feature_extractor, model_path, output_path, temp_feature_file='temp_features.csv'):
        """
        Use the trained model to make predictions on FASTA sequences

        Parameters:
        fasta_path: FASTA file path
        feature_extractor: Feature extractor, you must implement the extract(fasta_path, output_csv) method
        model_path: Model file path
        output_path: The file path to output the prediction results
        temp_feature_file: Intermediate Feature Save Path (default: 'temp_features.csv')
        """
        # Extract features
        feature_extractor.extract(fasta_path, temp_feature_file)

        # Read features
        X, seq_ids = [], []
        with open(temp_feature_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 2:
                    continue
                seq_id = parts[0]
                features = list(map(float, parts[1:]))
                X.append(features)
                seq_ids.append(seq_id)
        X = np.array(X)

        # Load the model and predict
        model = joblib.load(model_path)
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]

        # Output the result
        with open(output_path, 'w') as f:
            f.write("Sequence_ID,Predicted_Label,Probability\n")
            for seq_id, pred, prob in zip(seq_ids, y_pred, y_proba):
                f.write(f"{seq_id},{int(pred)},{prob:.4f}\n")
        print(f"The forecast results are saved to {output_path}")

    def plot_confusion_matrix(self, cm, labels, filename):
        """
        Create and save confusion matrix heat map

        Parameters:
        cm: Confusion matrix
        labels: Class label
        filename: The path to which the image is saved
        """
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.tight_layout()
        plt.savefig(filename, dpi=600)
        plt.close()

    def confusion_matrix_per_fold(self, X_train, y_train, X_test, y_test, class_names):
        """
        Confusion matrix plots for the validation set and the test set are generated separately on each fold

        Parameters:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        class_names: Category name
        """
        if self.best_model is None:
            raise ValueError("Call tune_and_validate() to get the best model first")

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_val, y_val = X_train[val_idx], y_train[val_idx]

            model = clone(self.best_model)
            model.fit(X_tr, y_tr)

            # Validation set
            val_pred = model.predict(X_val)
            val_cm = confusion_matrix(y_val, val_pred)
            self.plot_confusion_matrix(
                val_cm, class_names,
                f"Fold {fold} - Validation Set",
                filename=f"Figure/confusion_matrix_val_fold{fold}.png"
            )

            # Test set
            test_pred = model.predict(X_test)
            test_cm = confusion_matrix(y_test, test_pred)
            self.plot_confusion_matrix(
                test_cm, class_names,
                f"Fold {fold} - Test Set",
                filename=f"Figure/confusion_matrix_test_fold{fold}.png"
            )

    def plot_umap(self, X, y):
        """
        Normalize and UMAP dimensionality reduction and visualization of original features

        Parameters:
        X: Characteristic matrix
        y: Labels
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        reducer = umap.UMAP(n_components=2, random_state=42)
        X_umap = reducer.fit_transform(X_scaled)

        class_names = {1: 'high-secretion-efficiency', 0: 'low-secretion-efficiency'}

        plt.figure(figsize=(6, 5))

        for label in [1, 0]:
            plt.scatter(X_umap[y == label, 0], X_umap[y == label, 1],
                        label=class_names[label], alpha=0.7)

        plt.legend(loc='upper right')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')

        plt.tick_params(axis='both', direction='in')

        plt.tight_layout()
        plt.savefig("Figure/umap_raw_features.png", dpi=600)
        plt.close()

    def plot_umap_after_model(self, X, y, model_path, output_path="Figure/umap_after_model.png"):
        """
        Extract the hidden layer output of the model and visualize the dimensionality reduction of UMAP

        Parameters:
        X: Original feature
        y: Labels
        model_path: Saved model path
        output_path: Image save path (default: "Figure/umap_after_model.png")
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)

            model = joblib.load(model_path)
            if not isinstance(model, Pipeline):
                raise ValueError("The loaded model must be a sklearn Pipeline object")

            scaler = model.named_steps['scaler']
            mlp = model.named_steps['mlp']

            X_scaled = scaler.transform(X)

            # Extract hidden layer output
            def get_hidden_output(model, X_input):
                hidden_layer_sizes = model.hidden_layer_sizes
                if isinstance(hidden_layer_sizes, int):
                    hidden_layer_sizes = (hidden_layer_sizes,)

                activations = [X_input]
                for i in range(len(hidden_layer_sizes)):
                    z = np.dot(activations[i], model.coefs_[i]) + model.intercepts_[i]
                    if model.activation == 'relu':
                        a = np.maximum(0, z)
                    elif model.activation == 'tanh':
                        a = np.tanh(z)
                    elif model.activation == 'logistic':
                        a = 1 / (1 + np.exp(-z))
                    else:
                        raise ValueError(f"Unsupported activation functions: {model.activation}")
                    activations.append(a)
                return activations[-1]  # Output of the last hidden layer

            hidden_features = get_hidden_output(mlp, X_scaled)

            reducer = umap.UMAP(n_components=2, random_state=42)
            X_umap = reducer.fit_transform(hidden_features)

            plt.figure(figsize=(6, 5))
            class_names = {1: 'high-secretion-efficiency', 0: 'low-secretion-efficiency'}
            for label in [1, 0]:
                plt.scatter(X_umap[y == label, 0], X_umap[y == label, 1],
                            label=class_names[label], alpha=0.7)
            plt.legend(loc='upper right')
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
            plt.tick_params(axis='both', direction='in')
            plt.tight_layout()
            plt.savefig(output_path, dpi=600)
            plt.close()
