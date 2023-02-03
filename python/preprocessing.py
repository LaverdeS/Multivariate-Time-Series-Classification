import logging
import pathlib
import json
import pandas as pd


logging.basicConfig(level=logging.INFO)


def json_data_to_dataframe(path:str='.'):
    """Read all json files from path and returns a DataFrame"""

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
    
    
def detect_and_remove_blinking_from(df, columns:list=[]):
  """From some target keys, remove the 0.0 values from the time-series if any found"""
  
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
    blinks_p = len([d for ser in df.pupil_dilation for d in ser if d==0.0])
    blinks_b = len([d for ser in df.baseline for d in ser if d==0.0])
    df.pupil_dilation = df.pupil_dilation.map(lambda ser: [f for f in ser if f != 0.0])
    df.baseline = df.baseline.map(lambda ser: [f for f in ser if f != 0.0])
    logging.info("blinking values have been removed!")
  else:
    logging.info("no blinking values were found in your data!")
  logging.info("consider running outlier detection to clean your data!")

  logging.info(f"number of data points in pupil_dilation {number_of_data_points_p}")
  logging.info(f"number of data points in baseline: {number_of_data_points_b}")
  logging.info(f"number of blinks removed from pupil_dilation: {blinks_p}, {(blinks_p/number_of_data_points_p)*100}%")
  logging.info(f"number of blinks removed from baseline: {blinks_b}, {(blinks_b/number_of_data_points_b)*100}%")
  return df