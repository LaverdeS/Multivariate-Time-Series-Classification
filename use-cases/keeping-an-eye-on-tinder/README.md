<h1 align="center">
  Keeping an eye on Tinder
</h1>
<h3 align="center">
  Towards Automated Detection of Partner Selection via Pupillary Data from Eye-tracker and Smartphone Cameras
</h3>

### *Time-Series Explorations* Interactive App
The following web application demonstrates how to use this repository tools for rapid and interactively explore and clean, prepare and visualize your data, and to train a ROCKET model for univariate time-series classification.

[![App](https://huggingface.co/spaces/laverdes/ts-explorations)](https://laverdes-ts-explorations.hf.space)


### Time-Series Classification
Building upon the recent success of convolutional neural networks for time-series classification, we applied
the state-of-the-art ROCKET model (Random Convolutional Kernel Transform) which features a high
classification accuracy by transforming time series on basis of random convolutional kernels and training a
simple linear classifier [22]. Data was split using Stratisfied Kfold (k=5) to ensure a balanced proportion of the
labels, resulting in 102 samples to train ROCKET and 26 to test it (13 for each label). Given the small data
sets we selected the evaluation strategy LOO (LeaveOneOut) built in a Ridge Classifier. This allows for a
more complex analysis of how the model would perform when trained on unseen data including small
variations. Training was carried out five times with random sets to capture the best model (25 models trained
in total).
