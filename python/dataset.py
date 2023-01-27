import json
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import argparse
import random
import warnings

from preprocessing import normalize_data, normalize, normalize_single_lenght


float_formatter = "{:.4f}".format
np.set_printoptions(formatter={'float_kind': float_formatter})
warnings.filterwarnings("ignore", category=DeprecationWarning)

PARTICIPANTS = ["Anoth", "Arif", "Ashok", "Gowthom", "Josephin", "Raghu"]
DATA_PATH = ".data/"

CLASSIFIER_MODES = "pattern-agnostic", "pattern-specific", "pattern-agnostic-binary", "pattern-specific-binary"
CLASSIFIER_MODE = CLASSIFIER_MODES[0]
VERBOSE_LEVEL = 0


def read_json(path):
    with open(path, 'r') as file_in:
        return json.load(file_in)


def read_data(feature_selection="all"):
    if feature_selection not in ["distance", "pupil", "all"]:
        feature_selection = "all"
    distances = []
    pupils = []
    if feature_selection in ["distance", "all"]:
        distances = [read_json(DATA_PATH + participant_name + "Distance.txt") for participant_name in PARTICIPANTS]
    if feature_selection in ["pupil", "all"]:
        pupils = [read_json(DATA_PATH + participant_name + "Pupil.txt") for participant_name in PARTICIPANTS]
    if feature_selection == "distance":
        return distances
    if feature_selection == "pupil":
        return distances
    if feature_selection == "all":
        return distances, pupils


def build_dataframe(data_generator):
    confirm_pattern_matching = False
    if VERBOSE_LEVEL > 1:
        confirm_pattern_matching = True

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
    return df_concat


def print_bold(text): print(f"\033[1m {text} \033[0m")


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
        if ix == 1 and VERBOSE_LEVEL>1:
            print("\nsample: \n", sample)
        try:
            if type(sample['ts_pupil']) == list:
                distance_collection.append([sample['pattern_name'], sample['ts_distance']])
            else:
                distance_collection.append([sample['pattern_name'], sample['ts_distance'].tolist()])
        except AttributeError:
            print("error with: ", type(sample))

    return distance_collection


def get_pupil_in_dataframe(df, user_name):
    is_user = df['participant_name'] == user_name
    df = df[is_user]
    pupil_collection = []

    for ix, sample in df.iterrows():
        if ix == 1 and VERBOSE_LEVEL>1:
            print("\nsample: \n", sample)
        try:
            if type(sample['ts_pupil']) == list:
                pupil_collection.append([sample['pattern_name'], sample['ts_pupil']])
            else:
                pupil_collection.append([sample['pattern_name'], sample['ts_pupil'].tolist()])
        except AttributeError:
            print("error with: ", type(sample))

    return pupil_collection


def plot_collection(name: str, plot_only_this: str, collection,
                    number_of_desired_plots=0):
    """this method takes the distances_collection for one user and a specific
        pattern plot_only_this and add traces in the same graph until the
        number_of_desired_plots is reached; if 0 then plot all"""

    # todo: there is a lot of redundancy

    fig = go.Figure()
    plots_count = 0
    print("total number of graphs: ", len(collection))

    for (pattern_name, d) in collection:
        if pattern_name == plot_only_this:
            if plots_count == 0:
                fig.add_trace(go.Line(y=d))
                fig.update_layout(title=plot_only_this + " " + name)
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


def generate_data_structure(save_raw=False):
    # read distances and pupil dilatino info for each participant: list of lists(pattern_name, list)
    distances_data, pupils_data = read_data(feature_selection="all")

    # add the pattern_name information to the pupil_data(*_p)
    pupils_data = [add_pattern_name_to_ts(d, p) for d, p in zip(distances_data, pupils_data)]

    # build the Dataframe
    data_generator = [distances_data, PARTICIPANTS, pupils_data]
    df_concat = build_dataframe(data_generator)
    df_concat_no_norm = df_concat.copy(deep=True)

    if save_raw:
        save_dataframe(df_concat_no_norm, name="raw")
        print("raw data stored in: ", DATA_PATH)

    if "binary" in CLASSIFIER_MODE:
        # todo: add binary mode once the demo works
        pass

    # multi-participant classifier

    # data cleanining (blinking)

    # normalization by value and lenght
    df_concat, max_lenght = normalize_data(df_concat, by_value=True, by_lenght=True, verbose=VERBOSE_LEVEL > 0)
    patterns = list(set(df_concat.pattern_name.tolist()))
    print("\npattern names: ", patterns)
    return df_concat, patterns, df_concat_no_norm, max_lenght


def save_dataframe(df, name="norm"):
    df.to_csv(DATA_PATH + name + '_dataframe.csv', index=False)

    try:
        with pd.ExcelWriter(DATA_PATH + name + "_dataframe.xlsx", engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="norm_data")

        # df.to_excel('raw_dataframe.xls', index=False)
    except:
        print("skipping .xlsx format storage...")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--classifier", help="select the classification-task mode", choices=CLASSIFIER_MODES,
                        default="pattern-agnostic")
    parser.add_argument("-d", "--data_path", help="main data folder", type=str, default=DATA_PATH)
    parser.add_argument("-p", "--plot",
                        help="plot a sample of the participant original data and a single experiment with normalization",
                        action="store_true")
    parser.add_argument("-s", "--save", help="store the dataframe in .data", action="store_true")
    parser.add_argument("-v", "--verbosity", help="define the verbosity level", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    DATA_PATH = args.data_path
    CLASSIFIER_MODE = args.classifier
    VERBOSE_LEVEL = args.verbosity

    print("\nnumber of participants: ", len(PARTICIPANTS))
    print("participants_names: ", PARTICIPANTS)
    print("data_path: ", DATA_PATH)
    print("classifier: ", CLASSIFIER_MODE)
    print("verbosity_level: ", VERBOSE_LEVEL, "\n")

    data_df, patterns, data_df_no_norm, max_leght = generate_data_structure(save_raw=VERBOSE_LEVEL>1)  # return normalized data_df

    if args.verbosity > 1:
        print("\nraw:\n", data_df_no_norm.head(10))

    if args.verbosity > 0:
        print("\nnormal:\n", data_df.head(10))

    if args.save:
        save_dataframe(data_df)
        print_bold(f"\nnormalized data stored in: {DATA_PATH}\n")

    if args.plot:
        pattern = random.choice(patterns)
        participant = random.choice(PARTICIPANTS)
        plot_type = random.choice(["distance_data", "pupil_data"])
        data = None

        if plot_type == "distance_data":
            data = get_distance_in_dataframe(data_df, participant)
            print(f"plotting distance_data for {participant}-{pattern}")
        elif plot_type == "pupil_data":
            data = get_pupil_in_dataframe(data_df, participant)

        single_data = random.choice(
            get_pupil_in_dataframe(data_df_no_norm, participant)) if plot_type == "pupil_data" else random.choice(
            get_distance_in_dataframe(data_df_no_norm, participant))

        print(f"plotting {plot_type} for {participant}-{single_data[0]}")
        fig = plot_experiment(name=f"{plot_type} for {participant}-{single_data[0]}", l=single_data[1], normal=True, stretched=True, max_lenght=max_leght)
        fig.show(renderer="browser")

        fig = plot_collection(name=f"{plot_type} for {participant}-{pattern}", plot_only_this=pattern, collection=data)
        fig.show(renderer="browser")

    print_bold("\ndone...")


# todo: make more general method to include arbitrary features
# note: when running on terminal is possible to save and plot; when running in pycharm: get default arguments values
