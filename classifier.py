import numpy as np
import matplotlib.pyplot as plt
import argparse
import pytorch_lightning as pl
import seaborn as sns
import warnings

from multiprocessing import cpu_count
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import classification_report, confusion_matrix

from dataset import generate_data_structure
from model import UserDataModule, UserPredictor
from feature_engineering import feature_selection


pl.seed_everything(42)

float_formatter = "{:.4f}".format
np.set_printoptions(formatter={'float_kind': float_formatter})
warnings.filterwarnings("ignore", category=DeprecationWarning)

# global variables
CLASSIFIER_MODES = "pattern-agnostic", "pattern-specific", "pattern-agnostic-binary", "pattern-specific-binary"
FEATURES = ["pattern_name", "trace_difference", "pupil_dilation"]
FEATURE_COLUMNS = FEATURES[1:]  # this only works for the pattern agnostic
CLASSIFIER_MODE = CLASSIFIER_MODES[0]
VERBOSE_LEVEL = 0


def print_yellow(text): print("\033[93m {}\033[00m".format(text))


def print_bold(text): print(f"\033[1m {text} \033[0m")


def show_confusion_matrix(confusion_matrix):
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('True User')
    plt.xlabel('Predicted User')


def argparse():
    pass


if __name__ == '__main__':
    print_bold("\n______________________________________________________________________\n\
    LSTM TIME-SERIES MULTIVARIATE CLASSIFIER\n")
    # colors: https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal

    data_path = ".data/"
    print("data_path: ", data_path)

    PARTICIPANTS = ["Anoth", "Arif", "Ashok", "Gowthom", "Josephin", "Raghu"]

    # read data or generate it, dataset selection:
    data_df, *_ = generate_data_structure()

    # feature selection
    train_sequences, val_sequences, test_sequences, FEATURE_COLUMNS, label_encoder, pattern_encoder = \
        feature_selection(df_concat=data_df)

    print("train, val, test")
    len(train_sequences), len(val_sequences), len(test_sequences)

    # build and train the model

    N_EPOCHS = 10
    BATCH_SIZE = 4  # 8 for binary pattern-specific

    data_module = UserDataModule(train_sequences, test_sequences, val_sequences, BATCH_SIZE)

    # model
    print("cpu_count: ", cpu_count())

    model = UserPredictor(
        n_features=len(FEATURE_COLUMNS),
        n_classes=len(label_encoder.classes_)
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )

    logger = TensorBoardLogger("logs", name="usergaze")

    trainer = pl.Trainer(
        logger=logger,
        callbacks=checkpoint_callback,
        max_epochs=N_EPOCHS,
        accelerator="cpu",  # gpu
        devices=1,
        enable_progress_bar=True,
        log_every_n_steps=1
    )

    trainer.fit(model, data_module)

    # todo: tensorboards connection from local
