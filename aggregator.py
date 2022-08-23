import os
import json
import argparse
import sys
from os import walk
from tqdm import tqdm

DATA_IN_PATH = '.data/coords/'
# DATA_IN_PATH = 'C:/Users/lavml/Desktop/Freelance/TS/.data/pupil/'
DATA_OUT_PATH = '.data/'
FIELDS = ['gaze_valid', 'both_pupils_valid']


def read_json(path):
    with open(path, 'r') as file_in:
        return json.load(file_in)


def save_json(data, name, ending: str):
    with open(DATA_OUT_PATH + name + ending, 'w') as f:
        json.dump(data, f)


def extract_data_for_user(name=''):
    filenames = next(walk(DATA_IN_PATH + name), (None, None, []))[2]
    filenames = [f for f in filenames if "DS_Store" not in f]
    if "GazeData" in filenames[0]:
        samples_ix = [int(s.replace(".txt", "").replace("GazeData", "")) for s in filenames]
    else:
        samples_ix = [int(s.replace(".txt", "").replace("PupilData", "")) for s in filenames]
    sorter_dict = {name: ix for name, ix in zip(filenames, samples_ix)}
    sorter_dict = dict(sorted(sorter_dict.items(), key=lambda item: item[1]))
    filenames = sorter_dict.keys()
    return filenames


def read_filenames_to_list_of_lists(filenames, name, field: str):
    user_data = []
    user_data_x, user_data_y = [], []
    for filename in tqdm(filenames):
        if filename == '.DS_Store':
            continue
        d_data = read_json(DATA_IN_PATH + name + '/' + filename)['Items']
        series = []
        series_x, series_y = [], []
        if field == 'gaze_valid':
            for sample in d_data:
                try:
                    sample_f = sample[field]
                except KeyError:
                    print("______________________________data path error_______________________________", sample)
                    continue
                # print(sample_f)
                sample_x = sample_f['x']
                sample_y = sample_f['y']
                series_x.append(sample_x)
                series_y.append(sample_y)
            user_data_x.append(series_x)
            user_data_y.append(series_y)
        else:
            for sample in d_data:
                series.append(sample[field])
            user_data.append(series)
    if field == 'gaze_valid':
        return user_data_x, user_data_y
    else:
        return user_data


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--data_in", help="the path from where the raq data will be read.", default=DATA_IN_PATH)
    parser.add_argument("-o", "--data_out", help="the path to where the data aggregation will be stored",
                        default=DATA_OUT_PATH)
    parser.add_argument("-f", "--field", help="the field name inside the json output from the device", choices=FIELDS,
                        default="gaze_valid")
    return parser.parse_args()


if __name__ == '__main__':
    # notince: each feature has to be in separate user name fodler and independed txt files for it to work without key error
    args = parse_arguments()
    DATA_IN_PATH = args.data_in
    DATA_OUT_PATH = args.data_out

    print("reading from: ", DATA_IN_PATH)
    print("field: ", args.field)

    # for participants names get all subdirectory names for DATA_IN_PATH...
    participants = next(walk(DATA_IN_PATH), (None, None, []))[1]
    print()
    for ix, participant_name in enumerate(participants):
        print(participant_name)
        filenames = extract_data_for_user(participant_name)
        try:
            x, y = read_filenames_to_list_of_lists(filenames, participant_name, args.field)
            save_json(x, participant_name + '_x', 'coord.txt')
            save_json(y, participant_name + '_y', 'coord.txt')
        except ValueError:
            data = read_filenames_to_list_of_lists(filenames, participant_name, args.field)
            save_json(data, participant_name + 'Pupil', '.txt')


# arif second gaze sample pattern 4 [0] is 0.273157 not 0.263006!!!
