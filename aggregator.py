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


def save_json(data, name, ending:str):
    with open(DATA_OUT_PATH + name + ending, 'w') as f:
        json.dump(data, f)


def extract_data_for_user(name=''):
    filenames = next(walk(DATA_IN_PATH + name), (None, None, []))[2]
    return filenames


def read_filenames_to_list_of_lists(filenames, name, field:str):
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
    parser.add_argument("-o", "--data_out", help="the path to where the data aggregation will be stored", default=DATA_OUT_PATH)
    parser.add_argument("-f", "--field", help="the field name inside the json output from the device", choices=FIELDS, default="gaze_valid")
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
    for participant_name in participants:
        print(participant_name)
        filenames = extract_data_for_user(participant_name)
        x, y = read_filenames_to_list_of_lists(filenames, participant_name, args.field)
        save_json(x, participant_name + '_x', 'coord.txt')
        save_json(y, participant_name + '_y', 'coord.txt')

    # Arif_filenames = extract_data_for_user('Arif')
    # arif = read_filenames_to_list_of_lists(Arif_filenames, 'Arif', 'both_pupils_valid')
    #
    # Ashok_filenames = extract_data_for_user('Ashok')
    # ashok = read_filenames_to_list_of_lists(Ashok_filenames, 'Ashok', 'both_pupils_valid')
    #
    # Gowthom_filenames = extract_data_for_user('Gowthom')
    # gowthom = read_filenames_to_list_of_lists(Gowthom_filenames, 'Gowthom', 'both_pupils_valid')
    #
    # Josephin_filenames = extract_data_for_user('Josephin')
    # josephin = read_filenames_to_list_of_lists(Josephin_filenames, 'Josephin', 'both_pupils_valid')
    #
    # Raghu_filenames = extract_data_for_user('Raghu')
    # raghu = read_filenames_to_list_of_lists(Raghu_filenames, 'Raghu', 'both_pupils_valid')
    #
    # save_json(anoth, 'Anoth', 'Pupil.txt')
    # save_json(anoth, 'Arif', 'Pupil.txt')
    # save_json(anoth, 'Ashok', 'Pupil.txt')
    # save_json(anoth, 'Gowthom', 'Pupil.txt')
    # save_json(anoth, 'Josephin', 'Pupil.txt')
    # save_json(anoth, 'Raghu', 'Pupil.txt')

    # todo: generalize, create method to walk the folder and get participant_names
