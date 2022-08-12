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
    print("max_lenght: ", max_lenght)
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


if __name__ == '__init__':
    print_bold("\nLSTM TIME-SERIES MULTIVARIATE CLASSIFIER\n")

    data_path = ".data/"
    print("data_path: ", data_path)

    PARTICIPANTS = ["Anoth", "Arif", "Ashok", "Gowthom", "Josephin", "Raghu"]

    # read distances_collection for each participant: list of lists(pattern_name, list)
    anoth_distance = read_json("AnothDistance.txt")
    arif_distance = read_json("ArifDistance.txt")
    ashok_distance = read_json("AshokDistance.txt")
    gowthom_distance = read_json("GowthomDistance.txt")
    josephin_distance = read_json("JosephinDistance.txt")
    raghu_distance = read_json("RaghuDistance.txt")

    # read pupil_dilation for each participant (dilation: diameter changes)
    anoth_pupil = read_json("AnothPupil.txt")
    arif_pupil = read_json("ArifPupil.txt")
    ashok_pupil = read_json("AshokPupil.txt")
    gowthom_pupil = read_json("GowthomPupil.txt")
    josephin_pupil = read_json("JosephinPupil.txt")
    raghu_pupil = read_json("RaghuPupil.txt")

    # add the pattern_name information to the pupil_data(*_p)
    anoth_pupil = add_pattern_name_to_ts(anoth_distance, anoth_pupil)
    arif_pupil = add_pattern_name_to_ts(arif_distance, arif_pupil)
    ashok_pupil = add_pattern_name_to_ts(ashok_distance, ashok_pupil)
    gowthom_pupil = add_pattern_name_to_ts(gowthom_distance, gowthom_pupil)
    josephin_pupil = add_pattern_name_to_ts(josephin_distance, josephin_pupil)
    raghu_pupil = add_pattern_name_to_ts(raghu_distance, raghu_pupil)

    # Build the Dataframe
    data_generator = [
        [anoth_distance, arif_distance, ashok_distance, gowthom_distance, josephin_distance, raghu_distance],
        ["Anoth", "Arif", "Ashok", "Gowthom", "Josephin", "Raghu"],
        [anoth_pupil, arif_pupil, ashok_pupil, gowthom_pupil, josephin_pupil, raghu_pupil]]

    confirm_pattern_matching = False

    print("number of collections (participants): ", len(data_generator[0]), end="\n\n")

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
        print(f"{len(participant_data_distance)} rows are being been added for {participant_name}...")
        df_concat = pd.concat([df_concat, df_distance], ignore_index=True) if ix > 0 else df_distance

    print(f"\nDataFrame has now {len(df_concat.ts_distance)} samples...")

    df_concat = df_concat[["participant_name", "pattern_name", "ts_distance", "ts_pupil"]]
    print(df_concat.head(), df_concat.tail())


