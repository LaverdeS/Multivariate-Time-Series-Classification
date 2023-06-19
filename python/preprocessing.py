import logging
import pathlib
import json
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)


def json_data_to_dataframe(path: str = '.'):
    """
    Read all json files from path and returns a DataFrame where each row
    corresponds to the readings and labels of one experiment
    """

    logging.info(f"number of files: {len(list(pathlib.Path(path).glob('*.json')))}")
    logging.info(f"building base dataframe...\n")
    data_dict = dict()

    for json_file in pathlib.Path(path).glob('*.json'):
        with open(json_file, 'r') as file_in:
            data = json.load(file_in)
            for k in data:
                for i in data[k]:
                    for key, v in i.items():
                        if key not in list(data_dict.keys()):
                            data_dict[key] = [v]
                        else:
                            data_dict[key].append(v)

    return pd.DataFrame.from_dict(data_dict)


def min_listoflists_length(input_l: list = []):
    """
    From an input list composed of lists, get the maximum
    length of all the inner lists. Can be applied to more
    collections/iterables types.
    """
    min_length = 999999
    for i in input_l:
        if min_length > len(i):
            min_length = len(i)
    return min_length


def max_listoflists_length(input_l: list = []):
    """
    From an input list composed of lists, get the maximum
    length of all the inner lists. Can be applied to more
    collections/iterables types.
    """
    max_length = 0
    for i in input_l:
        try:
            if max_length < len(i):
                max_length = len(i)
        except TypeError:
            logging.warning(f"element in list: {i}")
            return None
    logging.debug(f"min_length: {min_listoflists_length(input_l)}")
    logging.debug(f"max_length: {max_length}")
    return max_length


def standardize(input_l: list = []):
    """
    Apply scaling method to make the values of an input list
    centered around mean with a unit standard deviation
    """
    output_l = (input_l - np.mean(input_l)) / np.std(input_l)
    output_l = [i.tolist() for i in output_l]
    logging.debug(f"max: {max(output_l)}")
    logging.debug(f"mean: {np.mean(output_l)}")
    logging.debug(f"std: {np.std(output_l)}")
    return output_l


def normalize_lengths(input_l: list = [], max_length=0):
    """
    From an input list composed of lists, normalize the lengths of the
    inner lists to the maximum length between all of them. Can be applied
    to more collections/iterables types.
    """
    if max_length == 0:
        max_length = max_listoflists_length(input_l)
    logging.info(f"max_length: {max_length}")
    new_l = [np.interp(np.linspace(0, 1, max_length).astype('float'),
                       np.linspace(0, 1, len(l_i)).astype('float'), l_i)
             for l_i in input_l]
    logging.debug(f"min: {str(min_listoflists_length(input_l))}")
    return new_l


def normalize_float_resolution_ts(df, columns: list, n_decimal: int = 4):
    """
    From an DataFrame, normalize the float values inside the time-series for the
    targeted columns list to a defined number of n_decimal's.
    """
    for column in columns:
        df[column] = [[round(float(fl), n_decimal) for fl in series] for series in
                      df[column].tolist()]
    return df


def normalize_float_resolution(df, columns: list, n_decimal: int = 4):
    """
    From an DataFrame, normalize the float values inside the targeted columns list
    to a defined number of decimals.
    """
    for column in columns:
        df[column] = [round(float(m), n_decimal) for m in df[column].tolist()]
    return df


def add_relative_to_baseline(column: str, df, baseline_column='baseline'):
    """
    Adds a relative calculated field by substracting each step value in the
    column target time series by the mean of the 'baseline' reading.
    """
    df['baseline_mean'] = [sum(s) / len(s) for s in df[baseline_column]]
    df[f'relative_{column}'] = [df[column][i] - df['baseline_mean'][i] for i in range(len(df))]
    return df
