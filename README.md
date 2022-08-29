<h1>
 <p align="center">
  <img width="300" height="300" src=".media/GazeDefy.png">
</p>
 </h1>

<h4 align="center">Multivariate Time Series Classification for Gaze Tracking Data</h4>

<p align="center">
 <a href="https://www.python.org/downloads/release/python-370/">
      <img src="https://img.shields.io/badge/python-3.7-brightgreen.svg">
  </a>
 <a href="https://saythanks.io/to/lavmlk20201">
      <img src="https://img.shields.io/badge/Say%20Thanks-!-1EAEDB.svg">
  </a>
 <a href="https://www.buymeacoffee.com/lavmlk2020B" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffee" height=22 width=110>
  </a>
</p>
 
The purpose of this repository is to provide the guidelines and a framework for time-series exploration and classification using eye-tracking data: **gaze shift** and **pupil diameter changes**. Even though the presented results are particular to the experimental setup, pipeline was design with scalability to other multivariate time-series classification tasks and other datasets. 

The task:
<p align="center">
<i><b>"To create a system capable of classifying users by finding an "eye-signature" in eye-tracking data".</b></i>
</p>

The data collection experiment: 
<p align="center">
<i><b>"Each participant will have to draw each of the 6 patterns 15 times with their gaze movement".</b></i>
</p>

The 6 patterns are composed by 3 lines each. The form shapes such as 'Z', 'N', 'U'. If Such a system could be used for security applications for instance. The "eye-signature" would be the PIN to identify a person and grant access.

<p align="center">
  <img width="700" height="390" src=".media/experimental_setup.png">
</p>
<p align="center">
  <sub>source: https://www.tobii.com/group/about/this-is-eye-tracking/
  </p>
  
### Overview: Pipeline description for classifying time series data
<sup>source: [IBM/what-is-time-series-classification](https://developer.ibm.com/learningpaths/get-started-time-series-classification-api/what-is-time-series-classification/)</sup>

## Requirements
Python version 3.7 or superior is strongly recommended. The requirements can be found inside `requirements.txt` and can be install running the command:
 
```{python}
!pip install --quiet -r requirements.txt
```

## How to use this repository?
Some sample data can be found inside the `.data` folder and the original plots of the data inside `.media`. The folowing is a short description of each of the `*.py` files inside this repository:

- `aggregator.py`: Every single experiment's output signals for a single user drawing one pattern is originally stored on a separate `*.txt` file.  This script reads and aggregates each single file inside a parcipant's data-folder into a single file with the structure:

```{python}
 [['pattern_name', List(time_series)]]
 ```

This is done for both the information about `ts_distance` and `ts_pupil`.
  
- `dataset.py`: dataset construction options for several classifier types...
  
- `preprocessing.py`: preprocessing...
  
- `feature_engineering.py`: feature_engineering...

- `classifier.py`: This script contains the general pipeline for reading the data, transforming it, fitting a ML model against it, and evaluating its performance.

#### Example execution
- Run `aggregator.py` and store data in .data
- Run `classifier.py by` selecting the dataset, the classifier_type, the model and dataset and the hyperparameters configuration.
- todo: `evaluate.py`

## Exploratory Data Analysis (EDA)
In total, 30 people participated in the test (data collection) but for the sake of cleaner visualizations, the notebooks and the plots that are shown here are using only data of 7 participants. The classifications reports and other outputs are for models that were trained using the data of these 6 participants as well.
 
Originally, the data is composed by time-series gaze-shift and pupil-diameter readings which are the output of the eye-tracking device during the experiments:
 
<h1>
<p align="center">
 <img width="300" height="300" src="tree/main/.media/original_dataframe.PNG">
</p>
</h1>

## Algorithms Overview
The following are the most common approaches for time-series classification. In **bold**: currently available models.
 - Distance-based approaches
 - Shapelet
 - Model Ensembles
 - Dictionary approaches
 - Interval-based approaches
 - **Deep Learning**: LSTM, CNN_Rocket
