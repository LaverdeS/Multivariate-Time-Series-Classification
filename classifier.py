import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import json
import plotly.graph_objects as go
import pandas as pd
import torch
import pytorch_lightning as pl
import seaborn as sns
import copy
import warnings

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from os import cpu_count, environ

import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.modules import dropout

from multiprocessing import cpu_count
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.functional import accuracy
from sklearn.metrics import classification_report, confusion_matrix

pl.seed_everything(42)

float_formatter = "{:.4f}".format
np.set_printoptions(formatter={'float_kind': float_formatter})
warnings.filterwarnings("ignore", category=DeprecationWarning)

# global variables
CLASSIFIER_MODES = "pattern-agnostic", "pattern-specific", "pattern-agnostic-binary", "pattern-specific-binary"
CLASSIFIER_MODE = CLASSIFIER_MODES[0]
VERBOSE_LEVEL = 0
PLOT = False


def print_yellow(text): print("\033[93m {}\033[00m".format(text))


def print_bold(text): print(f"\033[1m {text} \033[0m")


def read_json(path):
    with open(path, 'r') as file_in:
        return json.load(file_in)


def max_listoflists_lenght(l, verbose=False):
    """l is a list of lists"""
    max_lenght = 0
    for i in l:
        if max_lenght < len(i):
            max_lenght = len(i)
    if verbose:
        print("min_lenght: ", min_listoflists_lenght(l))
        print("max_lenght: ", max_lenght)
    return max_lenght


def min_listoflists_lenght(l):
    """l is a list of lists"""
    min_lenght = 999999
    for i in l:
        if min_lenght > len(i):
            min_lenght = len(i)
    return min_lenght


def normalize(l, verbose=False):
    l = (l - np.mean(l)) / np.std(l)
    l = [i.tolist() for i in l]
    if verbose:
        print("max: ", max(l))
        print("mean: ", np.mean(l))
        print("std: ", np.std(l))
    return l


def normalize_lenghts(l, verbose=False, max_lenght=0):
    """l is a list of lists"""
    if max_lenght == 0:
        max_lenght = max_listoflists_lenght(l, verbose)
    new_l = [np.interp(np.linspace(0, 1, max_lenght).astype('float'), np.linspace(0, 1, len(l_i)).astype('float'), l_i)
             for l_i in l]
    if verbose:
        min_listoflists_lenght(l)
    return new_l


def normalize_single_lenght(l, max_lenght=0):
    """l is a list"""
    if max_lenght == 0:
        max_lenght = max([len(i) for i in l])
    print("max_lenght: ", max_lenght)
    new_l = np.interp(np.linspace(0, 1, max_lenght).astype('float'), np.linspace(0, 1, len(l)).astype('float'), l)
    return new_l


def add_pattern_name_to_ts(distance_ts, pupil_ts):
    new_pupil_ts = []
    for distance, pupil in zip(distance_ts, pupil_ts):
        pattern_name, _ = distance
        new_pupil_ts.append([pattern_name, pupil])
    return new_pupil_ts


def get_distance_in_dataframe(df, user_name):
    is_user = df['participant_name'] == user_name
    df = df[is_user]
    distance_collection = []

    for ix, sample in df.iterrows():
        distance_collection.append([sample['pattern_name'], sample['ts_distance'].tolist()])

    return distance_collection


def get_pupil_in_dataframe(df, user_name):
    is_user = df['participant_name'] == user_name
    df = df[is_user]
    pupil_collection = []

    for ix, sample in df.iterrows():
        pupil_collection.append([sample['pattern_name'], sample['ts_pupil'].tolist()])

    return pupil_collection


def plot_collection(plot_only_this: str, distances_collection,
                    number_of_desired_plots=0):
    """this method takes the distances_collection for one user and a specific
        pattern plot_only_this and add traces in the same graph until the
        number_of_desired_plots is reached; if 0 then plot all"""

    # todo: there is a lot of redundancy, you could clear a lot of lines

    fig = go.Figure()
    plots_count = 0
    print("total number of graphs: ", len(distances_collection))

    for (pattern_name, d) in distances_collection:
        if pattern_name == plot_only_this:
            if plots_count == 0:
                fig.add_trace(go.Line(y=d))
                fig.update_layout(title=plot_only_this)
            else:
                fig.add_trace(go.Line(y=d))
            plots_count += 1
            fig.update_layout(width=1200, height=800)
            if number_of_desired_plots:
                if plots_count == number_of_desired_plots:
                    break

    print("plots: ", plots_count)
    return fig


def plot_experiment(name: str, l, normal=False, normal_only=False, stretched=False, max_lenght=0, verbose=False):
    if verbose:
        print("raw: ")
        print("max: ", max(l))
        print("mean: ", np.mean(l))
        print("std: ", np.std(l), end="\n\n")

    fig = go.Figure()
    fig.update_layout(title=name)
    fig.update_layout(width=1200, height=800)
    if not normal_only:
        fig.add_trace(go.Line(name="raw", y=l))

    if normal:
        l_normal = normalize(l, verbose=verbose)
        fig.add_trace(go.Line(name="nomalized_value", y=l_normal))

    if stretched:
        if max_lenght != 0:
            l_normal = normalize_single_lenght(l, max_lenght=max_lenght)
        else:
            l_normal = normalize_single_lenght(l)
        fig.add_trace(go.Line(name="nomalized_lenght", y=l_normal))

    return fig


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

    data_path = ".data/"
    print("data_path: ", data_path)

    PARTICIPANTS = ["Anoth", "Arif", "Ashok", "Gowthom", "Josephin", "Raghu"]

    # read distances_collection for each participant: list of lists(pattern_name, list)
    anoth_distance = read_json(data_path + "AnothDistance.txt")
    arif_distance = read_json(data_path + "ArifDistance.txt")
    ashok_distance = read_json(data_path + "AshokDistance.txt")
    gowthom_distance = read_json(data_path + "GowthomDistance.txt")
    josephin_distance = read_json(data_path + "JosephinDistance.txt")
    raghu_distance = read_json(data_path + "RaghuDistance.txt")

    # read pupil_dilation for each participant (dilation: diameter changes)
    anoth_pupil = read_json(data_path + "AnothPupil.txt")
    arif_pupil = read_json(data_path + "ArifPupil.txt")
    ashok_pupil = read_json(data_path + "AshokPupil.txt")
    gowthom_pupil = read_json(data_path + "GowthomPupil.txt")
    josephin_pupil = read_json(data_path + "JosephinPupil.txt")
    raghu_pupil = read_json(data_path + "RaghuPupil.txt")

    # add the pattern_name information to the pupil_data(*_p)
    anoth_pupil = add_pattern_name_to_ts(anoth_distance, anoth_pupil)
    arif_pupil = add_pattern_name_to_ts(arif_distance, arif_pupil)
    ashok_pupil = add_pattern_name_to_ts(ashok_distance, ashok_pupil)
    gowthom_pupil = add_pattern_name_to_ts(gowthom_distance, gowthom_pupil)
    josephin_pupil = add_pattern_name_to_ts(josephin_distance, josephin_pupil)
    raghu_pupil = add_pattern_name_to_ts(raghu_distance, raghu_pupil)

    # build the Dataframe
    data_generator = [
        [anoth_distance, arif_distance, ashok_distance, gowthom_distance, josephin_distance, raghu_distance],
        ["Anoth", "Arif", "Ashok", "Gowthom", "Josephin", "Raghu"],
        [anoth_pupil, arif_pupil, ashok_pupil, gowthom_pupil, josephin_pupil, raghu_pupil]]

    confirm_pattern_matching = False

    print("number of participants: ", len(data_generator[0]), end="\n")
    print("\nPARTICIPANTS: ", PARTICIPANTS)

    df_concat = pd.DataFrame()

    for ix, (participant_data_distance, participant_name, participant_data_pupil) in enumerate(
            zip(data_generator[0], data_generator[1], data_generator[2])):

        df_distance = pd.DataFrame(participant_data_distance, columns=["pattern_name", "ts_distance"])
        df_pupil = pd.DataFrame(participant_data_pupil, columns=["pattern_name", "ts_pupil"])

        if ix == 0 and confirm_pattern_matching:
            print("cofirming pattern...")
            print(df_distance['pattern_name'].head())
            print(df_pupil['pattern_name'].head())
            print()

        # put everything in one dataframe: df_distance then concat
        df_distance["ts_pupil"] = df_pupil["ts_pupil"]
        df_distance["participant_name"] = participant_name
        if VERBOSE_LEVEL > 0:
            print(f"{len(participant_data_distance)} rows are being been added for {participant_name}...")
        df_concat = pd.concat([df_concat, df_distance], ignore_index=True) if ix > 0 else df_distance

    print(f"\nDataFrame has {len(df_concat.ts_distance)} samples...")

    df_concat = df_concat[["participant_name", "pattern_name", "ts_distance", "ts_pupil"]]

    if "binary" in CLASSIFIER_MODE:
        pass

    # multi-participant classifier
    # normalization
    if VERBOSE_LEVEL > 0:
        print("checking sequence lenghts matching (first 20)...\n")

        for ix, (d, p) in enumerate(zip(df_concat['ts_distance'], df_concat['ts_pupil'])):
            print(f"lenght of sequence in ts_distance: {len(d)}, lenght of sequence in ts_pupil: {len(p)}", end=" ")
            print_bold(len(d) == len(p))
            if ix == 20:
                break

    print("\nmin lenght in ts_distance = ", min_listoflists_lenght(df_concat['ts_distance'].tolist()))
    print("min lenght in ts_pupil = ", min_listoflists_lenght(df_concat['ts_pupil'].tolist()))
    print("max lenght in ts_distance = ", max_listoflists_lenght(df_concat['ts_distance'].tolist()))
    print("max lenght in ts_pupil = ", max_listoflists_lenght(df_concat['ts_pupil'].tolist()))

    # apply normalization to the DataFrame (value and lenght)
    df_concat.ts_distance = df_concat.ts_distance.apply(normalize)
    df_concat.ts_pupil = df_concat.ts_pupil.apply(normalize)
    print("\nthe data has been normalized by value, mean=0, std=1")

    df_concat.ts_distance = normalize_lenghts(df_concat.ts_distance.tolist())
    df_concat.ts_pupil = normalize_lenghts(df_concat.ts_pupil.tolist(),
                                           max_lenght=max_listoflists_lenght(df_concat['ts_distance']))
    print("the data has been normalized by lenght (stretched to the max lenght)")

    if VERBOSE_LEVEL > 0:
        print("\nmin lenght in ts_distance = ", min_listoflists_lenght(df_concat['ts_distance'].tolist()))
        print("min lenght in ts_pupil = ", min_listoflists_lenght(df_concat['ts_pupil'].tolist()))
        print("max lenght in ts_distance = ", max_listoflists_lenght(df_concat['ts_distance'].tolist()))
        print("max lenght in ts_pupil = ", max_listoflists_lenght(df_concat['ts_pupil'].tolist()))

    if PLOT:
        gowthom_distance = get_distance_in_dataframe(df_concat, "Gowthom")
        fig = plot_collection("pattern5", gowthom_distance)
        fig.show(renderer="browser")

    # feature selection --> explode the DataFrame ts_distance and ts_pupil

    for ix, series in tqdm(df_concat.iterrows(), total=len(df_concat)):
        series_df_distance = pd.DataFrame(series.ts_distance, columns=["trace_difference"])
        series_df_pupil = pd.DataFrame(series.ts_pupil, columns=["pupil_dilation"])

        series_df_distance["pupil_dilation"] = series_df_pupil['pupil_dilation']
        series_df_distance["participant_name"] = series.participant_name
        series_df_distance["pattern_name"] = series.pattern_name
        series_df_distance["series_id"] = ix
        series_df_distance = series_df_distance[
            ["participant_name", "pattern_name", "series_id", "trace_difference", "pupil_dilation"]]
        X = series_df_distance if ix == 0 else pd.concat([X, series_df_distance])

    # filter dfs selecting the desired features

    FEATURES = ["pattern_name", "trace_difference", "pupil_dilation"]
    PREDICTION_TARGET = "participant_name"

    y = X[["series_id", "pattern_name", "participant_name"]]
    y = y[["series_id", PREDICTION_TARGET]]

    X_columns = ["series_id"]
    X_columns.extend(FEATURES)
    X = X[X_columns]

    y.reset_index(inplace=True)
    del y["index"]
    X.reset_index(inplace=True)
    del X["index"]

    print(f"\nX: {len(X)}; y: {len(y)}; equal? {len(X) == len(y)}")

    if VERBOSE_LEVEL > 0:
        print(
            f"\nConsidering {len(PARTICIPANTS)} participants with {len(anoth_distance)} experiments each, we have {len(PARTICIPANTS) * len(anoth_distance)} number of samples or single graphs.\n\
        Naturally the lenght of the time-series for a single experiment is dependent on the speed of the user at that pattern attempt. \n\
        The absolute minimum lenght for this data collection is {min_listoflists_lenght(df_concat.ts_distance.tolist())} and the maximum {max_listoflists_lenght(df_concat.ts_distance.tolist())}. \n\
        After normalization, all time series have the same lenght: {max_listoflists_lenght(df_concat.ts_distance.tolist())}. \n\
        Therefore, each equal-lenght time series (one single row) can be exploded into {max_listoflists_lenght(df_concat.ts_distance.tolist())} rows. \n\
        In order to identify the original row index before exploding, a new column 'series_id' is added to de dataframe. \n\
        Ergo, the total number of samples for my new dataframes X and y is {len(PARTICIPANTS) * len(anoth_distance) * max_listoflists_lenght(df_concat.ts_distance.tolist())}.")

        print("data class distribution after transformation...\n")
        print(y.participant_name.value_counts())
        print("\n")
        y.participant_name.value_counts().plot(kind="bar")
        plt.xticks(rotation=45)

        print("data class distribution after transformations...\n")
        print(X.pattern_name.value_counts())
        print("\n")
        X.pattern_name.value_counts().plot(kind="bar")
        plt.xticks(rotation=45)

    # encode pattern_name using LabelEncoder()

    pattern_encoder = LabelEncoder()

    encoded_patterns = pattern_encoder.fit_transform(X.pattern_name)
    print({k: v for k, v in zip(pattern_encoder.classes_.tolist(), set(encoded_patterns))})

    X["pattern_id"] = encoded_patterns
    X.reset_index(inplace=True)
    del X["index"]

    if "specific" in CLASSIFIER_MODE:
        pass

    del X['pattern_name']
    del X['pattern_id']

    FEATURE_COLUMNS = X.columns.tolist()[1:]

    # encode labels: participant_name

    label_encoder = LabelEncoder()

    encoded_labels = label_encoder.fit_transform(y.participant_name)
    print({k: v for k, v in zip(label_encoder.classes_.tolist(), set(encoded_labels))})

    y["label"] = encoded_labels
    y.reset_index(inplace=True)
    del y["index"]

    #### train-test-val-split

    # train, val, test split stratifies (balanced)

    targets = []
    data = []
    for series_id, group in X.groupby("series_id"):
        sequence_features = group[FEATURE_COLUMNS]
        label = y[y.series_id == series_id].iloc[0].label  # all series would have same label so just take one
        data.append(sequence_features)
        targets.append(label)

    train_sequences, test_sequences, ytrain, ytest = train_test_split(data, targets, test_size=0.2, random_state=42,
                                                                      stratify=targets)
    train_sequences, val_sequences, ytrain, yval = train_test_split(train_sequences, ytrain, test_size=0.2,
                                                                    random_state=42, stratify=ytrain)

    train_sequences = [(sequence, label) for sequence, label in zip(train_sequences, ytrain)]
    test_sequences = [(sequence, label) for sequence, label in zip(test_sequences, ytest)]
    val_sequences = [(sequence, label) for sequence, label in zip(val_sequences, yval)]

    print("train, val, test")
    len(train_sequences), len(val_sequences), len(test_sequences)


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

        def pad_sequences(self):
            pass


    print("cpu_count: ", cpu_count())


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

            #### BUILD MODEL AND TRAIN IT


    N_EPOCHS = 800
    BATCH_SIZE = 16  # 8 for binary pattern-specific

    data_module = UserDataModule(train_sequences, test_sequences, val_sequences, BATCH_SIZE)


    # model

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


    model = UserPredictor(
        n_features=len(FEATURE_COLUMNS),
        n_classes=len(label_encoder.classes_)
    )

    # init tensorboar

    # %load_ext tensorboard
    # %tensorboard --logdir ./logs

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=False,
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

    # !wget -O mini.sh https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh
    # !chmod +x mini.sh
    # !bash ./mini.sh -b -f -p /usr/local
    # !conda install -q -y jupyter
    # !conda install -q -y google-colab -c conda-forge
    # !python -m ipykernel install --name "py39" --user

    # environ['WANDB_CONSOLE'] = 'off'
    trainer.fit(model, data_module)