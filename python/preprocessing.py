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

    logging.info(f"number of files: {len(list(pathlib.Path().glob('*.json')))}")
    logging.info(f"building base dataframe...\n")
    data_dict = dict()

    for json_file in pathlib.Path('.').glob('*.json'):
        with open(json_file, 'r') as file_in:
            data = json.load(file_in)
            for k in data:
                for i in data[k]:
                    for k, v in i.items():
                        if k not in list(data_dict.keys()):
                            data_dict[k] = [v]
                        else:
                            data_dict[k].append(v)

    return pd.DataFrame.from_dict(data_dict)


def detect_and_remove_blinking_from(df, columns: list = []):
    """
    From some target keys, remove the 0.0 values from the time-series if any found.
    This method assumes there are 'baseline' and 'pupil_dilation' columns in the data.
    """

    remove_blinking = False
    for ts_key in columns:
        for ser in df[ts_key]:
            for f in ser:
                if f == 0.0:
                    remove_blinking = True

    for ser in df['baseline']:
        for f in ser:
            if f == 0.0:
                remove_blinking = True

    if remove_blinking:
        logging.info("blinking (0.0) values were found in the data!")
        number_of_data_points_p = len([d for ser in df.pupil_dilation for d in ser])
        number_of_data_points_b = len([d for ser in df.baseline for d in ser])
        blinks_p = len([d for ser in df.pupil_dilation for d in ser if d == 0.0])
        blinks_b = len([d for ser in df.baseline for d in ser if d == 0.0])
        df.pupil_dilation = df.pupil_dilation.map(lambda ser: [f for f in ser if f != 0.0])
        df.baseline = df.baseline.map(lambda ser: [f for f in ser if f != 0.0])
        logging.info("blinking values have been removed!")
    else:
        logging.info("no blinking values were found in your data!")
    logging.info("consider running outlier detection to clean your data!")

    logging.info(f"number of data points in pupil_dilation {number_of_data_points_p}")
    logging.info(f"number of data points in baseline: {number_of_data_points_b}")
    logging.info(
        f"number of blinks removed from pupil_dilation: {blinks_p}, {(blinks_p / number_of_data_points_p) * 100}%")
    logging.info(f"number of blinks removed from baseline: {blinks_b}, {(blinks_b / number_of_data_points_b) * 100}%")
    return df


def min_listoflists_length(input_l):
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


def max_listoflists_length(input_l):
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


def normalize_lengths(input_l, max_length=0):
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


def add_relative_to_baseline(column: str, df):
    """
    Adds a relative calculated field by substracting each step value in the
    column target time series by the mean of the 'baseline' reading.
    """
    df['baseline_mean'] = [sum(s) / len(s) for s in df['baseline']]
    df[f'relative_{column}'] = [df[column][i] - df['baseline_mean'][i] for i in range(len(df))]
    return df
