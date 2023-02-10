<h3 align="center">
  <img
    src="https://user-images.githubusercontent.com/38184042/215089990-e4d34345-46f8-4280-ba39-42e13b19a9f1.png"
    height="200"
  >
</h3>

<div align="center">

  <a href="https://github.com/Unstructured-IO/unstructured/blob/main/LICENSE.md">![https://pypi.python.org/pypi/unstructured/](https://img.shields.io/pypi/l/unstructured.svg)</a>
  <a href="https://www.python.org/downloads/release/python-370/"><img src="https://img.shields.io/badge/python-3.7-brightgreen.svg"></a>
  <a href="https://github.com/Unstructured-IO/unstructured/blob/main/CODE_OF_CONDUCT.md">![code_of_conduct.md](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg) </a>
  <a href="https://pypi.python.org/pypi/unstructured/">![https://github.com/Naereen/badges/](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)</a>
<a href="https://www.buymeacoffee.com/lavmlk2020B" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffee" height=22 width=110>
  </a>
</div>

<h1 align="center">
 Multivariate-Time-Series-Classification
</h1>

The purpose of this repo is to provide some tools for time-series [Exploratory Data Analysis (EDA)](https://www.ibm.com/topics/exploratory-data-analysis) and data preparation pipelines for machine learning applications and research with eye-tracking data: gaze and pupil dilation in. The initial processing and transformation blocks enhance the researcher for rapid-prototyping data applications and first-hand data cleaning, visualization and chained transformations. 

The tool-box is organized by modules found on the [python](python) folder. The tools are part of one of the following families:
- Preprocessing tools: including data loader, DataFrame constructures, transformation functions to format, standarize, and normalize the data.
- Visualizing tools: plotting methods that assist in EDA of time series data and reporting.
- Purging tools: methods use to clean data points from time-series features and to detect + visualize + remove outliers from the data by statistical methods such as [Median Absolute Deviation (MAD)](https://www.graphpad.com/support/faq/what-is-the-median-absolute-deviation-mad-/) and [Interquartile Range (IQR)](https://statisticsbyjim.com/basics/interquartile-range/).

## :coffee: Getting Started

* Create a [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-python.html) to work in and activate it, e.g. conda environment named 'ts-tools':

	`conda create ts-tools python=3.7` <br />
	`conda activate ts-tools`
	
* Install the requirements using the `requirements.txt` and python package manage pip:
	
	`pip install -r requirements.txt`
	
## :clap: Quick Tour

You can run this [Jupyter notebook](notebooks/Time-Series-EDA-tinder.ipynb) to quickly check some methods that served for one of the use-cases: [Keeping an eye on Tinder](https://github.com/LaverdeS/Multivariate-Time-Series-Classification/tree/main/use-cases/keeping-an-eye-on-tinder)

<!-- See the [documentation](https://github.com/LaverdeS/Multivariate-Time-Series-Classification) for a full description
of the features in the library. -->

The following code summarizes the how to use chained methods from the tool-box for doing a clean EDA and data preparation in 6 steps. This input for this pipeline are the `.json` files and the output is a `.csv` containing ml-ready data. This is the equivalent of a data pipeline and some methods for visualizing time-series data:

```python
from python.preprocessing import json_data_to_dataframe, add_relative_to_baseline, 
from python.preprocessing import normalize_lengths, normalize_float_resolution_ts, standarize
from python.visualizing import plot_collection
from python.purging import remove_outliers_mad_single_feature

# Load and purge the data from blinking values
df = json_data_to_dataframe(path='sample-data/tinder')
df = detect_and_remove_blinking_from(df, ['pupil_dilation', 'baseline'])

# Visualize the data
pupil_collection = [(rating, series_i) for rating, series_i in zip(df.rating, df.pupil_dilation)]
HTML(plot_collection(4, pupil_collection).to_html())

# Add calculated fields and normalize lengths
df = add_relative_to_baseline('pupil_dilation', df)
df['relative_pupil_dilation'] = normalize_lengths(df.relative_pupil_dilation.tolist())

# Remove extreme outliers using MAD, normalize float resolution and standarize time-series
df = remove_outliers_mad_single_feature(df, column='relative_pupil_dilation')
df = normalize_float_resolution_ts(df, columns=['pupil_dilation', 'relative_pupil_dilation', 'baseline'])
df.relative_pupil_dilation = df.relative_pupil_dilation.apply(standarize)

# Visualize the transformed data
relative_pupil_collection = [(rating, series_i) for rating, series_i in zip(df.rating, df.relative_pupil_dilation)]
HTML(plot_collection(4, relative_pupil_collection).to_html())

# Save to disk
df.to_csv("ml-ready-data.csv")
```

### 🛠️ Tools / Blocks
The developer tools can be found inside the [python](https://github.com/LaverdeS/Multivariate-Time-Series-Classification/tree/main/python) directory.

## 💼 Use Cases
The following examples are using the tools provided by this repository and can be foundational for similar kind of work.
- [Keeping an Eye on Tinder: ](https://github.com/LaverdeS/Multivariate-Time-Series-Classification/tree/main/use-cases/keeping-an-eye-on-tinder) Towards Automated Detection of Partner Selection via Pupillary Data from Eye-tracker and Smartphone Cameras
- [Eye-D:](https://github.com/LaverdeS/Multivariate-Time-Series-Classification/tree/main/use-cases/eye-d) Identifying Users by their Gaze and Pupil Diameter Data while Drawing Patterns

## 📝 License

The [GNU General Public License](https://github.com/LaverdeS/Multivariate-Time-Series-Classification/blob/main/LICENSE): Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed. Note that free here refers to freedom but not free of price. Doing this repository took several hours. This time and effort is with the spirit of providing the research community with beneficial tools for their eye-tracking projects. Everyone is welcome to contribute. If you find this repository useful and want to suppot the author, you can [Buy Me a Coffe!](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)

<!--
## :books: Learn more

| Section | Description |
|-|-|
| [text tag](url) | description |
| [text](url) | description |
-->
