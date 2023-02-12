import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from warnings import simplefilter
from sktime.datatypes._panel._convert import from_nested_to_2d_array
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from python.visualizing import show_confusion_matrix

logging.basicConfig(level=logging.INFO)


class BinaryTimeSeriesClassifier(object):
    """
    Classifier object definition. This class deals with loading the data,
    creating training sets and training/tuning the defined model/method
    """
    def __init__(self, data_path: str, label_column: str, method: str, multi: bool = False) -> None:
        """
        Classifier constructor. It prepares the data for processing/developing from path
        """
        self.labels = [label_column, None]
        self.data = self._load_data_from_path(data_path)
        self.mode = "univariate" if not multi else "multivariate"
        self.method = method
        """
        Extra config/attributes can be added to set the training strategy. For instance:
        setting the classifier that the tabularization training branch will use and wether to display figure while training or not
        """

    def _prepare_binary_labels(self, df, label_column: str, positive_threshold):
        """
        Mapping function to assert the float|int label_column is suited for binary classification
        and to transform it to re-label it to binary int [0, 1].
        """
        df[label_column] = df[label_column].map(lambda i: 1 if i > positive_threshold else 0)
        self.labels[1] = df[self.labels[0]]
        return df

    def _load_data_from_path(self, path: str):
        """
        Data loader method to get the data from path and prepare dataframe structure
        with time-series as list type and the labels column as int str, this method
        also sets the second value for the self.labels tuple.
        """
        df = pd.read_csv(path)
        logging.info(f"number of samples: {df.size}")
        df = self._prepare_binary_labels(df, self.labels[0], positive_threshold=3)
        if 'Unnamed: 0' in df.keys():
            df.index = df['Unnamed: 0'].tolist()
            del df['Unnamed: 0']
        if 'baseline' in df.keys():
            df.baseline = df.baseline.apply(lambda i: json.loads(i))
        df.pupil_dilation = df.pupil_dilation.apply(lambda i: json.loads(i))
        df.relative_pupil_dilation = df.relative_pupil_dilation.apply(lambda i: json.loads(i))
        df[self.labels[0]] = df[self.labels[0]].apply(lambda i: str(int(i)))
        self.labels[1] = df[self.labels[0]]
        return df

    def build_training_data(self, X_column: str):
        """
        This methods created X and y dataframes depending if the intended use is for
        univariate or multivariate classification
        """
        if self.mode == 'univariate':
            self.X = self.data.loc[:, [X_column]]
            self.y = self.labels[1]
            self.X = self.X.applymap(lambda s: pd.Series(s))
            self.data = pd.concat([self.X, self.y], axis=1)
            assert self.X.size == self.y.size, "X and y shapes don't match, check your data."
        elif self.mode == 'multivariate':
            logging.warning(
                "This module needs to be updated to use this functionality")  # works trivial for tabularization

    def save_training_data(self, filename):
        """
        Filename is the absolute or relate path and filename to save to file;
        it must include the .csv extension
        """
        assert hasattr(self, 'X'), "This classifier has no training data yet You need to call build_training_data first"
        self.data.to_csv(filename)

    def train(self, k_folds: int = 5):
        """
        Training cycle definition and execution; make imports on demand to avoid overtime
        It uses StratifiedKFolds strategy to obtain the best model perfomance for the
        test data. When the dataset is small,
        """
        assert hasattr(self, 'X'), "This classifier has no training data yet You need to call build_training_data first"
        simplefilter(action='ignore', category=FutureWarning)
        seeds = [42, 1, 999, 4276, 67534]
        n_samples = min(self.data.rating.value_counts())
        best_global_acc = 0
        best_model = pd.DataFrame()
        logging.info(f"number of samples per class: {n_samples}")
        logging.info(f"tuning to obtain the best model (method: {self.method}, k_folds: {k_folds})...")

        if self.method.lower() == "tabularization":
            """
            use a RandomForestClassifier as default, but it can be easily 
            adapted to use ANY other classifier type after tabularization is done, including rocket.
            """
            for seed in seeds:
                df_sample = self.data.groupby('rating', group_keys=False).apply(
                    lambda x: x.sample(n_samples, random_state=seed))
                y = df_sample['rating']
                X = df_sample
                del X['rating']
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                print("\n_____________________________________________________________________")

                best_acc = 0
                best_df_eval = pd.DataFrame()

                for ix, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
                    print("\nfold: ", ix)
                    X_train, X_test = X.take(train_index), X.take(test_index)
                    y_train, y_test = y.take(train_index), y.take(test_index)
                    labels, counts = np.unique(y_train, return_counts=True)
                    print(labels, counts)

                    n_display = 5
                    print("displaying first {n_display} samples for each class")
                    fig, ax = plt.subplots(1, figsize=plt.figaspect(0.25))

                    for i in range(n_display):
                        for label in labels:
                            X_train.loc[y_train == label, "relative_pupil_dilation"].iloc[i].plot(ax=ax,
                                                                                                  label=f"class {label}",
                                                                                                  color='r' if label == "0" else 'b')
                    plt.legend()
                    ax.set(title="Fold time series first", xlabel="Time");
                    ax.get_legend().remove()
                    plt.show()

                    X_train_tab = from_nested_to_2d_array(X_train)
                    X_test_tab = from_nested_to_2d_array(X_test)

                    classifier = DummyClassifier(strategy="prior")
                    classifier.fit(X_train_tab, y_train)
                    print("Dummy score: ", classifier.score(X_test_tab, y_test))

                    classifier = RandomForestClassifier(n_estimators=100)
                    classifier.fit(X_train_tab, y_train)
                    y_pred = classifier.predict(X_test_tab)
                    print("random forest score: ", accuracy_score(y_test, y_pred))

                    predictions = classifier.predict(X_test_tab)
                    df_eval = pd.DataFrame([(p, t) for p, t in zip(predictions, y_test.tolist())],
                                           columns=["prediction", "truth"])
                    if best_acc < classifier.score(X_test_tab, y_test):
                        best_acc = classifier.score(X_test_tab, y_test)
                        best_df_eval = df_eval

                print("\nbest acc: ", round(best_acc, 2))
                if best_acc > best_global_acc:
                    best_global_acc = best_acc
                    best_model = best_df_eval

            print("\nbest global acc: ", round(best_global_acc, 2))
            print(classification_report(best_model.truth.tolist(), best_model.prediction.tolist(),
                                        target_names=list(set(best_model.truth.tolist()))))
            cm = confusion_matrix(best_model.truth.tolist(), best_model.prediction.tolist())
            df_cm = pd.DataFrame(cm, index=list(set(best_model.truth.tolist())),
                                 columns=list(set(best_model.truth.tolist())))
            print("\nTABULARIZATION DATAFRAME SAMPLE: \n")
            print(X_train_tab.head())
            show_confusion_matrix(df_cm)
            plt.show()

        elif self.method.lower() == "rocket":
            from sktime.transformations.panel.rocket import MiniRocket, MiniRocketMultivariate
            pass
        elif self.method.lower() == "feature-extractor":
            pass
        else:
            logging.warning("Available time-serie classification methods: tabularization, rocket, feature-extractor")