import pandas as pd
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Simple torch ANN for classification
class ANNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hidden_layers):
        super(ANNClassifier, self).__init__()
        self.hidden_layers = hidden_layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        for _ in range(self.hidden_layers):
            out = self.fc(out)
            out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out
    
def train(
    model: nn.Module,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    learning_rate: float = 0.001,
    num_epochs: int = 1000,
    batch_size: int = 2048,
) -> nn.Module:
  
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.9, step_size=100)
    
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize early stopping variables
    best_loss = float("inf")
    best_model_weights = None
    patience = 100
    threshold = 1e-3

    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.6f}')
        
        if loss < best_loss - threshold:
            best_loss = loss
            best_model_weights = copy.deepcopy(model.state_dict())
            patience = 100
        else:   
            patience -= 1
            if patience == 0:
                print(f'Early stopping on epoch {epoch}')
                model.load_state_dict(best_model_weights)
                break
        
    return model
            
def evaluate(model: nn.Module, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    
    model.eval()
    
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32, device=device)
    y_pred = model(X_test_tensor)
    y_pred = y_pred.cpu().detach().numpy().round()

    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(f'Precision: {precision_score(y_test, y_pred)}')
    print(f'Recall: {recall_score(y_test, y_pred)}')
    print(f'F1: {f1_score(y_test, y_pred)}')
    print(f'MCC: {matthews_corrcoef(y_test, y_pred)}')
    