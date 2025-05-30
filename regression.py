from extract_features import Regression_FeatureExtractor
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
from torch.utils.data import TensorDataset, DataLoader


# ================= Feature extraction =================
regression_extractor = Regression_FeatureExtractor(
    model_file='checkpoints/esm2_t33_650M_UR50D.pt',
    regression_weights_file='checkpoints/esm2_t33_650M_UR50D-contact-regression.pt'
)
# Extract test set features
regression_extractor.extract(
    data_file='data-regression/test.csv',
    output_file='output-regression/test.csv'
)
# Extract the features of the training set
regression_extractor.extract(
    data_file='data-regression/train_fold_1.csv',
    output_file='output-regression/train_1.csv'
)
# Extract validation set features
regression_extractor.extract(
    data_file='data-regression/validation_fold_1.csv',
    output_file='output-regression/validation_1.csv'
)

# ================= Set up a random seed to ensure that the results are reproducible =================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ================= Data loading =================
# Load the extracted feature file
train_data = pd.read_csv('output-regression/train_1.csv')
validation_data = pd.read_csv('output-regression/validation_1.csv')
test_data = pd.read_csv('output-regression/test.csv')

# Convert data to Tensor format
X_train = torch.tensor(train_data.iloc[:, 1:].values, dtype=torch.float32)
y_train = torch.tensor(train_data.iloc[:, 0].values, dtype=torch.float32).unsqueeze(1)
X_validation = torch.tensor(validation_data.iloc[:, 1:].values, dtype=torch.float32)
y_validation = torch.tensor(validation_data.iloc[:, 0].values, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(test_data.iloc[:, 1:].values, dtype=torch.float32)
y_test = torch.tensor(test_data.iloc[:, 0].values, dtype=torch.float32).unsqueeze(1)

# ================= Model definition =================
# MLP is built for regression tasks
class MLPRegressor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLPRegressor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.net(x)

# ================= Hyperparameter configuration =================
# Input Feature Dimensions
input_size = X_train.shape[1]
# Number of neurons in the hidden layer
hidden_size = 64
# Maximum learning rate
max_lr = 5e-5
# Warm-up ratio
warmup_ratio = 0.1
# Total number of training rounds
epochs = 3000
# Batch size
batch_size = 32

# Build the training dataset and loader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

total_steps = len(train_loader) * epochs

# Instantiate models and optimizers
model = MLPRegressor(input_size, hidden_size)
optimizer = optim.AdamW(model.parameters(), lr=max_lr)

# Warm-up learning rate scheduler
def lr_lambda(current_step):
    if current_step < warmup_ratio * total_steps:
        return float(current_step) / (warmup_ratio * total_steps)
    return 1.0

scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

# ================= Training the model =================
criterion = nn.MSELoss()  # Mean square error as the loss function
model.train()

step = 0
for epoch in range(epochs):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            print(f"[Epoch {epoch+1} | Step {step}] Loss: {loss.item():.6f}, LR: {scheduler.get_last_lr()[0]:.8f}")
        step += 1

# ================= Validate model performance =================
model.eval()
with torch.no_grad():
    val_preds = model(X_validation).squeeze().numpy()
    val_true = y_validation.squeeze().numpy()

    r2 = r2_score(val_true, val_preds)
    mse = mean_squared_error(val_true, val_preds)
    pearson_corr, _ = pearsonr(val_true, val_preds)

    print("\n Validation set evaluation metrics：")
    print(f"R²: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"Pearson correlation coefficient: {pearson_corr:.4f}")

with torch.no_grad():
    test_preds = model(X_test).squeeze().numpy()
    test_true = y_test.squeeze().numpy()

    r2_test = r2_score(test_true, test_preds)
    mse_test = mean_squared_error(test_true, test_preds)
    pearson_test, _ = pearsonr(test_true, test_preds)

    print("\n The test set evaluates the metrics：")
    print(f"R²: {r2_test:.4f}")
    print(f"MSE: {mse_test:.4f}")
    print(f"Pearson correlation coefficient: {pearson_test:.4f}")

pd.DataFrame(test_preds).to_csv('test_predictions.csv', index=False)
