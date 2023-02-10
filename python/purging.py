import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt

from scipy import stats
from python.visualizing import plot_outliers_in

logging.basicConfig(level=logging.INFO)


def detect_and_remove_blinking_from(df, columns: list = []):
    """
    From some target keys, remove the 0.0 values from the time-series if any found.
    This method assumes there are 'baseline' and 'pupil_dilation' columns in the data.
    Note: the 0.0 value can be easily replaced by any arbitrary value or interval
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
    number_of_data_points_p, number_of_data_points_b, blinks_p, blinks_b = 0, 0, 0, 0
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


def mad_method(df, variable_name):
    """
    Outlier detection using the Median Absolute Deviation Takes two
    parameters: dataframe and a column name of interest as string.
    Returns list of index for outliers rows
    """
    columns = df.columns
    med = np.median(df, axis=0)
    mad = np.abs(stats.median_abs_deviation(df))
    threshold = 3
    outlier = []
    index = 0
    for item in range(len(columns)):
        if columns[item] == variable_name:
            index = item
    for i, v in enumerate(df.loc[:, variable_name]):
        t = (v - med[index]) / mad[index]
        if t > threshold:
            outlier.append(i)
        else:
            continue
    return outlier


def remove_outliers_mad_single_feature(df, column: str):
    """
    Detect and remove outliers found in the data by the Mean Absolute Deviation
    method applied to a single time-series column
    """
    df_new = df.copy(deep=True)
    outliers_count = {'0': 0, '1': 0}
    total_number_of_data_points_pre, total_number_of_data_points_post = 0, 0

    for ix, single in enumerate(df[column]):
        df_single = pd.DataFrame(single)
        df_single.columns = ['reading']
        total_number_of_data_points_pre += len(df_single)
        outlier_ids = mad_method(df_single, "reading")
        outliers_count['1'] += len(outlier_ids)
        update_df = df_single.drop([df_single.index[out_id] for out_id in outlier_ids])
        df_new[column][ix] = update_df['reading']

        if outlier_ids:
            if len(update_df) >= len(df_single):
                logging.critical("the removal went wrong...")

        outlier_ids = mad_method(update_df, "reading")
        if outlier_ids:
            pass
        total_number_of_data_points_post += len(update_df)
    logging.info(f"total number of data points in df: {total_number_of_data_points_pre}")
    logging.info(f"total number of outliers: {outliers_count['1']}")
    logging.info(f"total number of data points in df after outlier removal: {total_number_of_data_points_post}")

    if total_number_of_data_points_pre - total_number_of_data_points_post != outliers_count['1']:
        logging.critical("the removal went wrong...")

    number_of_data_points_rp = len([d for ser in df_new[column] for d in ser])
    no = len([d for ser in df[column] for d in ser])
    logging.info(
        f"percentaje of data points removed : {no - number_of_data_points_rp}, {((no - number_of_data_points_rp) / no) * 100}%")
    return df_new


def iqr_method(df, k=3, feature='', outliers_name=''):
    """
    The Interquartile range (IQR) is calculated as the difference between the 75th and the 25th
    percentiles of the data. To identify outliers define the limits on the sample values
    that are a factor k of the IQR below the 25th percentile or above the 75th percentile.
    The common value for the factor k is the value 1.5. A factor k of 3 or more can be used to
    identify values that are extreme outliers or 'far outs'.
    """
    q1_feature, q3_len_distance = df[feature].quantile([0.25, 0.75])
    iqr_pc = q3_len_distance - q1_feature
    lower_pc = q1_feature - (k * iqr_pc)
    upper_pc = q3_len_distance + (k * iqr_pc)
    df[outliers_name] = ((df[feature] > upper_pc) | (df[feature] < lower_pc)).astype('int')
    return df


def iqr_analysis(df, column, k=3, plot=True, purge=False):
    """
    Performs outlier count and gets the row id with max number of
    outliers and the max value. Returns tuple of (int, tuple(int,int))
    """
    outliers_count = {'0': 0, '1': 0}
    max_outlier = [0, 0]
    df_new = df.copy(deep=True)
    total_number_of_data_points_pre, total_number_of_data_points_post = 0, 0
    for ix, single in enumerate(df[column]):
        df_single = pd.DataFrame(single)
        df_single.columns = ['reading']
        total_number_of_data_points_pre += len(df_single)
        df_single = iqr_method(df_single, k, 'reading', 'outlier')
        outliers_count['0'] += dict(df_single.outlier.value_counts())[0]
        try:
            outliers_count['1'] += dict(df_single.outlier.value_counts())[1]
            if dict(df_single.outlier.value_counts())[1] > max_outlier[0]:
                max_outlier[0] = dict(df_single.outlier.value_counts())[1]
                max_outlier[1] = ix
            if plot:
                fig = plot_outliers_in(df_single, column)
                logging.info(f"experiment index: {ix}")
                plt.show(fig)
            if purge:
                df_single = df_single[df_single['outlier'] == 0]
                df_new['relative_pupil_dilation'][ix] = df_single['reading']
        except KeyError:
            pass
        finally:
            total_number_of_data_points_post += len(df_single)
    logging.info(f"outliers count: {outliers_count}")
    logging.info(f"max_outlier: {str(max_outlier)}")
    if purge:
        logging.info(f"total_number_of_data_points_pre: {total_number_of_data_points_pre}")
        logging.info(f"total_number_of_data_points_post: {total_number_of_data_points_post}")
        if total_number_of_data_points_pre - total_number_of_data_points_post != outliers_count['1']:
            logging.debug(
                f"{total_number_of_data_points_pre - total_number_of_data_points_post} != {outliers_count['1']}")
            raise Exception("Something went wrong with the removal...")
        return outliers_count, max_outlier, df_new
    return outliers_count, max_outlier, df


def count_outliers(df_target, column: str):
    """
    Helper function to perform only outliers count for the targeted column name
    """
    outliers_count = {'0': 0, '1': 0}
    for ix, single in enumerate(df_target[column]):
        df_single = pd.DataFrame(single)
        df_single.columns = ['reading']
        df_single = iqr_method(df_single, 3, 'reading', 'outlier')
        outliers_count['0'] += dict(df_single.outlier.value_counts())[0]
        try:
            outliers_count['1'] += dict(df_single.outlier.value_counts())[1]
        except KeyError:
            pass
    return outliers_count


def purge_iter_iqr_method(df, column: str, n_iter: int):
    """
    Iteratively purge the time-series data in 'column' using outliers
    removal by the IQR method
    """
    df_new_iter = df.copy(deep=True)
    outliers_count = {'0': 0, '1': 0}
    max_outlier = [0, 0]
    for i in range(n_iter):
        for ix, single in enumerate(df_new_iter[column]):
            df_single = pd.DataFrame(single)
            df_single.columns = ['reading']
            df_single = iqr_method(df_single, 3, 'reading', 'outlier')
            outliers_count['0'] += dict(df_single.outlier.value_counts())[0]
            try:
                outliers_count['1'] += dict(df_single.outlier.value_counts())[1]
                if dict(df_single.outlier.value_counts())[1] > max_outlier[0]:
                    max_outlier[0] = dict(df_single.outlier.value_counts())[1]
                    max_outlier[1] = ix
                df_single = df_single[df_single['outlier'] == 0]
                df_new_iter[column][ix] = df_single['reading']
            except KeyError:
                pass
        logging.info(f"_________new round {i}__________")
        logging.info(f"current: {count_outliers(df_new_iter, column)}")
    logging.info(f"outliers count: {outliers_count}")
    logging.info(f"max_outlier: {str(max_outlier)}")
    return df_new_iter
