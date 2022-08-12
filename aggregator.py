from os import walk
from tqdm import tqdm
import json


DATA_PATH = 'C:/Users/lavml/Desktop/Freelance/TS/.data/pupil/'


def read_json(path):
    with open(path, 'r') as file_in:
        return json.load(file_in)


def save_json(data, name):
    with open(DATA_PATH + name + 'Pupil.txt', 'w') as f:
        json.dump(data, f)


def extract_pupil_data_for_user(name=''):
    filenames = next(walk(DATA_PATH + name), (None, None, []))[2]
    return filenames


def read_filenames_to_list_of_lists(filenames, name):
    user_data = []
    for filename in tqdm(filenames):
        if filename == '.DS_Store':
            continue
        d_data = read_json(DATA_PATH + name + '/' + filename)['Items']
        series = []
        for sample in d_data:
            series.append(sample['both_pupils_valid'])
        user_data.append(series)
    return user_data


if __name__ == '__main__':

    Anoth_filenames = extract_pupil_data_for_user('Anoth')
    anoth = read_filenames_to_list_of_lists(Anoth_filenames, 'Anoth')

    Arif_filenames = extract_pupil_data_for_user('Arif')
    arif = read_filenames_to_list_of_lists(Arif_filenames, 'Arif')

    Ashok_filenames = extract_pupil_data_for_user('Ashok')
    ashok = read_filenames_to_list_of_lists(Ashok_filenames, 'Ashok')

    Gowthom_filenames = extract_pupil_data_for_user('Gowthom')
    gowthom = read_filenames_to_list_of_lists(Gowthom_filenames, 'Gowthom')

    Josephin_filenames = extract_pupil_data_for_user('Josephin')
    josephin = read_filenames_to_list_of_lists(Josephin_filenames, 'Josephin')

    Raghu_filenames = extract_pupil_data_for_user('Raghu')
    raghu = read_filenames_to_list_of_lists(Raghu_filenames, 'Raghu')

    save_json(anoth, 'Anoth')
    save_json(anoth, 'Arif')
    save_json(anoth, 'Ashok')
    save_json(anoth, 'Gowthom')
    save_json(anoth, 'Josephin')
    save_json(anoth, 'Raghu')
