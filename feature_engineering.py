import pandas as pd
import matplotlib.pyplot as plt
import argparse

from tqdm import tqdm
from preprocessing import max_listoflists_lenght, min_listoflists_lenght
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

PARTICIPANTS = ["Anoth", "Arif", "Ashok", "Gowthom", "Josephin", "Raghu"]
CLASSIFIER_MODES = "pattern-agnostic", "pattern-specific", "pattern-agnostic-binary", "pattern-specific-binary"
CLASSIFIER_MODE = CLASSIFIER_MODES[0]
DATA_PATH = '.data/norm_dataframe.csv'
VERBOSE_LEVEL = 1


def print_bold(text): print(f"\033[1m {text} \033[0m")


def str2list(l):
    l = l.replace("[", "")
    l = l.replace("]", "")
    l = l.replace("\n", "")
    l = l.split()
    return l


def feature_selection(df_concat):
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
            f"\nConsidering {len(PARTICIPANTS)} participants with {90} experiments each, we have {len(PARTICIPANTS) * 90} number of samples or single graphs. \
Naturally the lenght of the time-series for a single experiment is dependent on the speed of the user at that pattern attempt. \
The absolute minimum lenght for this data collection is {min_listoflists_lenght(df_concat.ts_distance.tolist())} and the maximum {max_listoflists_lenght(df_concat.ts_distance.tolist())}. \n\
After normalization, all time series have the same lenght: {max_listoflists_lenght(df_concat.ts_distance.tolist())}. \
Therefore, each equal-lenght time series (one single row) can be exploded into {max_listoflists_lenght(df_concat.ts_distance.tolist())} rows. \
In order to identify the original row index before exploding, a new column 'series_id' is added to de dataframe. \n\
Ergo, the total number of samples for my new dataframes X and y is {len(PARTICIPANTS) * 90 * max_listoflists_lenght(df_concat.ts_distance.tolist())}.\n")

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

    return train_sequences, test_sequences, val_sequences, FEATURE_COLUMNS, label_encoder, pattern_encoder


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbosity", help="define the verbosity level", type=int, default=1)
    parser.add_argument("-d", "--data_path", help="relative path of the .csv file", type=str, default=DATA_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    VERBOSE_LEVEL = args.verbosity
    DATA_PATH = args.data_path
    print(f"\nverbose_level: ", VERBOSE_LEVEL)
    print(f"reading data from {DATA_PATH}\n")

    df = pd.read_csv(DATA_PATH)
    if VERBOSE_LEVEL > 0:
        print("sample ts_distance prior[0:20]: ", end="")
        print(df.ts_distance.tolist()[0][0:20])

    df.ts_distance = df.ts_distance.apply(str2list)
    df.ts_pupil = df.ts_pupil.apply(str2list)

    if VERBOSE_LEVEL > 0:
        print("\nsample ts_distance post[0:20]: ", end="")
        print(df.ts_distance.tolist()[0][0:20])
        print()

    # if "raw" in data_
    train_sequences, test_sequences, val_sequences, *_ = feature_selection(df)
    print_bold("\ndone...")
