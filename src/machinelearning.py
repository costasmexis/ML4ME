import copy

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from skrules import SkopeRules
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

""" Deep Learning - Artificial Neural Network (ANN) """
# Simple torch ANN for classification
class ANNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hidden_layers):
        super().__init__()
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
    
def train_ann(
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
    threshold = 1e-6

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

def get_predictions(model: nn.Module, X: pd.DataFrame) -> np.ndarray:
    """ Get predictions from a trained model. """
    model.eval()
    X_tensor = torch.tensor(X.values, dtype=torch.float32, device=device)
    y_pred = model(X_tensor)
    y_pred = y_pred.cpu().detach().numpy().round()
    return y_pred

def evaluate_ann(model: nn.Module, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    
    model.eval()
    
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32, device=device)
    y_pred = model(X_test_tensor)
    y_pred = y_pred.cpu().detach().numpy().round()

    print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
    print(f'Precision: {precision_score(y_test, y_pred):.4f}')
    print(f'Recall: {recall_score(y_test, y_pred):.4f}')
    print(f'F1: {f1_score(y_test, y_pred):.4f}')
    print(f'MCC: {matthews_corrcoef(y_test, y_pred):.4f}')
    
           
""" Machine Learning """
def evaluate_sklearn(model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    
    X_test.columns = X_test.columns.astype(str)
    y_pred = model.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
    print(f'Precision: {precision_score(y_test, y_pred):.4f}')
    print(f'Recall: {recall_score(y_test, y_pred):.4f}')
    print(f'F1 score: {f1_score(y_test, y_pred):.4f}')
    print(f'ROC AUC score: {roc_auc_score(y_test, y_pred):.4f}')
    print(f'MCC: {matthews_corrcoef(y_test, y_pred):.4f}')


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, 
                  scoring: str = "accuracy", cv: int = 3,
                  n_trials: int = 100) -> xgb.XGBClassifier:
        
    def objective(trial):
        
        param_dist_xgb = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "n_estimators": trial.suggest_int("n_estimators", 10, 100),
            "max_depth": trial.suggest_int("max_depth", 2, 12),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            # "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
            "tree_method": trial.suggest_categorical("tree_method", ["auto", "exact", "approx", "hist"]),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 0.5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 1),
        }
        
        model = xgb.XGBClassifier(**param_dist_xgb)
        score = cross_val_score(model, X_train, y_train, scoring=scoring, cv=cv, verbose=10, n_jobs=-1).mean()
        return score

    study = optuna.create_study(direction="maximize")  
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    best_model = xgb.XGBClassifier(**best_params)
    best_model.fit(X_train, y_train)
    return best_model
    

def train_decisiontree(X_train: pd.DataFrame, y_train: pd.Series, 
                       scoring: str = "accuracy", cv: int = 3,
                       n_trials: int = 100) -> DecisionTreeClassifier:
        
    def objective(trial):
        
        param_dist = {
            "max_depth": trial.suggest_int("max_depth", 2, 25),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 25),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 25),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),            
        }
        
        model = DecisionTreeClassifier(**param_dist)
        score = cross_val_score(model, X_train, y_train, scoring=scoring, cv=cv, verbose=10, n_jobs=-1).mean()
        return score

    study = optuna.create_study(direction="maximize")  
    study.optimize(objective, n_trials=n_trials)    

    best_params = study.best_params
    best_model = DecisionTreeClassifier(**best_params)
    # Make X_train columns as string -> needed for `DecisionTreeClassifier.feature_names_in_`
    X_train.columns = X_train.columns.astype(str)
    best_model.fit(X_train, y_train)
    return best_model


def train_skoperules(X_train: pd.DataFrame, y_train: pd.Series,
                     scoring: str = "accuracy", cv: int = 3, n_iter: int = 50) -> SkopeRules:

    # Define the parameter grid
    param_dist = {
        'precision_min': [0.1, 0.2, 0.3],
        'recall_min': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [10, 20, 30, 50],
        'max_samples': [0.2, 0.4, 0.8],
        'max_depth_duplication': [2, None],
    }

    skope_rules_clf = SkopeRules(
        feature_names=X_train.columns.values.tolist(),
        random_state=42,
        n_jobs=-1
    )

    random_search = RandomizedSearchCV(
        skope_rules_clf, param_distributions=param_dist, scoring=scoring, n_iter=n_iter, cv=3, verbose=10, n_jobs=-1, random_state=42
    )
    random_search.fit(X_train, y_train)
    skope_rules_clf = random_search.best_estimator_
    return skope_rules_clf