import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
import numpy as np
import random

seed = 2
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

train_data = pd.read_csv('train_1.csv')
val_data = pd.read_csv('val_1.csv')
test_data = pd.read_csv('test.csv')

X_train = train_data.iloc[:, 1:].values
y_train = train_data['WA'].values

X_val = val_data.iloc[:, 1:].values
y_val = val_data['WA'].values

X_test = test_data.iloc[:, 1:].values
y_test = test_data['WA'].values

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )

    def forward(self, x):
        return self.layers(x)

def train_model(model, criterion, optimizer, X_train, y_train, X_val, y_val, epochs=1140):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs.squeeze(), y_train)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs.squeeze(), y_val)
                print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}, Val Loss: {val_loss.item()}')

input_size = X_train.shape[1]
model = MLP(input_size=input_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, criterion, optimizer, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor)

model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor).squeeze().numpy()
    test_r2 = r2_score(y_test, test_predictions)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    test_corr, _ = pearsonr(y_test, test_predictions)
    print(f'test R2 Score: {test_r2}')
    print(f'test RMSE: {test_rmse}')
    print(f'test Pearson Correlation: {test_corr}')

    val_predictions = model(X_val_tensor).squeeze().numpy()
    val_r2 = r2_score(y_val, val_predictions)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    val_corr, _ = pearsonr(y_val, val_predictions)
    print(f'val R2 Score: {val_r2}')
    print(f'val RMSE: {val_rmse}')
    print(f'val Pearson Correlation: {val_corr}')

