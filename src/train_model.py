import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)

# ============================================================
# Classical Baseline Models
# ============================================================

def train_svm(X_train, y_train, X_test, y_test):

    model = SVC(kernel="rbf", probability=True)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="macro", zero_division=0)
    rec  = recall_score(y_test, preds, average="macro", zero_division=0)
    f1   = f1_score(y_test, preds, average="macro", zero_division=0)

    cm   = confusion_matrix(y_test, preds)

    return model, acc, prec, rec, f1, cm


def train_decision_tree(X_train, y_train, X_test, y_test):

    model = DecisionTreeClassifier()

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="macro", zero_division=0)
    rec  = recall_score(y_test, preds, average="macro", zero_division=0)
    f1   = f1_score(y_test, preds, average="macro", zero_division=0)

    cm   = confusion_matrix(y_test, preds)

    return model, acc, prec, rec, f1, cm


# ============================================================
# Bidirectional RNN Block (Used in DAN)
# ============================================================

class BiRNNBlock(nn.Module):

    def __init__(self, input_dim, hidden_dim=64, dropout=0.3):
        super(BiRNNBlock, self).__init__()

        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            nonlinearity="tanh"
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        out, _ = self.rnn(x)

        return self.dropout(out)


# ============================================================
# Bidirectional LSTM Block
# ============================================================

class BiLSTMBlock(nn.Module):

    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):

        super(BiLSTMBlock, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x):

        out, _ = self.lstm(x)

        out = self.layer_norm(out)

        return self.dropout(out)


# ============================================================
# DAN Model
# ============================================================

class DAN(nn.Module):

    def __init__(self, input_dim, num_classes=2, dropout=0.3):

        super(DAN, self).__init__()

        self.birnn = BiRNNBlock(
            input_dim=input_dim,
            hidden_dim=64,
            dropout=dropout
        )

        self.bilstm = BiLSTMBlock(
            input_dim=128,
            hidden_dim=128,
            dropout=dropout
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):

        x = self.birnn(x)

        x = self.bilstm(x)

        embedding = x.mean(dim=1)

        logits = self.classifier(embedding)

        return logits, embedding


# ============================================================
# Standalone BiLSTM (Baseline)
# ============================================================

class StandaloneLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim=128,
                 num_layers=2, dropout=0.3, num_classes=2):

        super(StandaloneLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.norm = nn.LayerNorm(hidden_dim * 2)

        self.drop = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):

        out, _ = self.lstm(x)

        out = self.norm(out)

        emb = out.mean(dim=1)

        return self.fc(self.drop(emb))


# ============================================================
# Standalone BiRNN (Baseline)
# ============================================================

class StandaloneRNN(nn.Module):

    def __init__(self, input_dim, hidden_dim=128,
                 num_layers=2, dropout=0.3, num_classes=2):

        super(StandaloneRNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            nonlinearity="tanh",
            dropout=dropout if num_layers > 1 else 0
        )

        self.norm = nn.LayerNorm(hidden_dim * 2)

        self.drop = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):

        out, _ = self.rnn(x)

        out = self.norm(out)

        emb = out.mean(dim=1)

        return self.fc(self.drop(emb))