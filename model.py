import copy
import torch
import pytorch_lightning as pl
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse

from torch.nn.modules import dropout
from os import cpu_count, environ
from torch.utils.data import Dataset, DataLoader
from torchmetrics.functional import accuracy
from os import cpu_count


def init_weights(m, verbose=False):
    if "LSTM" in str(m):
        parameters = copy.deepcopy(m._parameters)
        for param, v in m._parameters.items():
            try:
                if "weight" in param:
                    parameters[param] = torch.nn.init.xavier_uniform_(v)
                elif "bias" in param:
                    parameters[param] = v.fill_(0.01)
            except RuntimeError:
                if verbose:
                    print(f"lstm_parameter {param} uses grad and cannot be initialized")
        print("LSTM module weights and biases initialized")
        return parameters
    torch.nn.init.xavier_uniform_(m.weight)
    m.bias.data.fill_(0.01)
    print("Linear module weights and biases initialized")


class UserDataset(Dataset):

    def __init__(self, sequences) -> None:
        super().__init__()
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence, label = self.sequences[idx]
        return dict(
            sequence=torch.Tensor(sequence.to_numpy()),
            labels=torch.tensor(label).long()
        )


class UserDataModule(pl.LightningDataModule):

    def __init__(self, train_sequences, test_sequences, val_sequences, batch_size):
        super().__init__()
        self.train_sequences = train_sequences
        self.test_sequences = test_sequences
        self.val_sequences = val_sequences
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = UserDataset(self.train_sequences)
        self.test_dataset = UserDataset(self.test_sequences)
        self.val_dataset = UserDataset(self.val_sequences)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=cpu_count()
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=cpu_count()
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=cpu_count()
        )


class SequenceModel(nn.Module):

    def __init__(self, n_features, n_classes, n_hidden=256, n_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.75  # hyperparameter for regularization
        )
        self.classifier = nn.Linear(n_hidden, n_classes)

        # self.lstm.apply(init_weights)
        # self.classifier.apply(init_weights)

    def forward(self, x):
        self.lstm.flatten_parameters()  # for multi_gpu purposes
        _, (hidden, _) = self.lstm(x)
        out = hidden[-1]
        return self.classifier(out)


class UserPredictor(pl.LightningModule):

    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.model = SequenceModel(n_features, n_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, labels=None):
        output = self.model(x)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["labels"]
        loss, outputs = self(sequences, labels)
        predictions = torch.argmax(outputs, dim=1)
        step_accuracy = accuracy(predictions, labels)

        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_accuracy", step_accuracy, prog_bar=True, logger=True)
        return {"loss": loss, "accuracy": step_accuracy}

    def validation_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["labels"]
        loss, outputs = self(sequences, labels)
        predictions = torch.argmax(outputs, dim=1)
        step_accuracy = accuracy(predictions, labels)

        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_accuracy", step_accuracy, prog_bar=True, logger=True)
        return {"loss": loss, "accuracy": step_accuracy}

    def test_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["labels"]
        loss, outputs = self(sequences, labels)
        predictions = torch.argmax(outputs, dim=1)
        step_accuracy = accuracy(predictions, labels)

        self.log("test_loss", loss, prog_bar=True, logger=True)
        self.log("test_accuracy", step_accuracy, prog_bar=True, logger=True)
        return {"loss": loss, "accuracy": step_accuracy}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.0001)