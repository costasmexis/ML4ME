import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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
