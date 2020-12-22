"""
GridSearchScikitLearn

Author: G Bettsworth

2020
"""

from random import seed
from time import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.dummy import DummyClassifier

from pandas import DataFrame, Series

from numpy import ndarray, genfromtxt

data_location = r"../binary_classifiers_comparison_data"
output_location = r"../binary_classifiers_comparison_outputs"

seed(123)

array_a = genfromtxt(rf"{data_location}/a_wh_question_datapoints.txt", skip_header=1)
target_a = genfromtxt(rf"{data_location}/a_wh_question_targets.txt")

array_b = genfromtxt(rf"{data_location}/b_wh_question_datapoints.txt", skip_header=1)
target_b = genfromtxt(rf"{data_location}/b_wh_question_targets.txt")

classifiers = [DummyClassifier(strategy='most_frequent'), DecisionTreeClassifier(), KNeighborsClassifier(),
               RandomForestClassifier(),
               MLPClassifier(), SVC(), AdaBoostClassifier(), GaussianProcessClassifier()]

scaler = StandardScaler()
train_set = array_a
test_set = array_b
train_targets = target_a
test_targets = target_b


# TODO - add scaler error logging


class GridSearchClassifier:
    """
    Tool to try different Scikit-Learn classifiers for a given classification task.

    Parameters
    -----------
    train_set: numpy.ndarray
        Training set.
    test_set: numpy.ndarray
        Test set.
    train_targets: numpy.ndarray
        Training targets (categories). This should a single series.
    test_targets: numpy.ndarray
        Test targets (categories).
    classifiers: list
        List of Scikit-Learn classifiers.
    weight: float, default=0.5
        Weight for ranking the importance of accuracy as a measure of performance.
        Time ranking is weighted by (1-weight).
    scaler: , default=StandardScaler()
        Scaler from the Scikit-Learn library.
    """

    def __init__(self, train_set, test_set, train_targets, test_targets, classifiers, weight=0.5,
                 scaler=StandardScaler()):
        """
        Error logging performed and variables set.
        """

        variables = [train_set, test_set, train_targets, test_targets, classifiers, weight]
        variable_names = ['train_set', 'test_set', 'train_targets', 'test_targets', 'classifiers', 'weight']
        types = [ndarray, ndarray, ndarray, ndarray, list, float]

        for variable, name, variable_type in zip(variables, variable_names, types):
            if not isinstance(variable, variable_type):
                raise TypeError(f"The variable {name} needs to be a {variable_type}")

        if weight > 1.0:
            raise ValueError(f"You have specified weight={weight}. This weight is too large,"
                             f" it must be less than or equal to 1 and greater than or equal to 0.")

        if weight < 0.0:
            raise ValueError(f"You have specified weight={weight}. This weight is too small,"
                             f" it must be greater than or equal to 0 and less than or equal to 1.")

        self.train_set = train_set
        self.test_set = test_set
        self.train_targets = train_targets
        self.test_targets = test_targets
        self.scaler = scaler
        self.classifiers = classifiers
        self.weight = weight

    def _pipeline(self, scaler, model):
        """
        Train-test pipeline including scaling.

        Parameters
        -----------
        scaler: Scikit-Learn scaler
        model: Scikit-Learn classifier

        Returns
        --------
        score: float
            Accuracy score on test set.
        time: float
            Time taken in seconds for pipeline.
        """

        t = time()

        fitted_scaler = scaler.fit(self.train_set)
        train_set = fitted_scaler.transform(self.train_set)
        test_set = fitted_scaler.transform(self.test_set)

        fitted_model = model.fit(X=train_set, y=train_targets)
        predictions = fitted_model.predict(test_set)

        score = accuracy_score(y_true=self.test_targets, y_pred=predictions)
        final_time = time() - t

        return score, final_time

    def fit(self):
        """
        Fit all classifiers in list to the train-test pipeline. Rank best performing on the basis of (weighted)
        accuracy and time.

        Returns
        --------
        df: pandas.DataFrame
            DataFrame ranking each algorithm in order. Algorithm name is the index. Accuracy, time and ranks are reported.
        """

        classifier_strings = [str(x) for x in classifiers]

        df = DataFrame(columns=['accuracy', 'time'], index=classifier_strings)

        for classifier in self.classifiers:
            print("--------------------------------------")
            print(f'Testing: {classifier}')
            score, time = self._pipeline(scaler=self.scaler, model=classifier)

            print(f'Accuracy: {score}')
            print(f'Time: {time}')
            print("--------------------------------------")

            df.loc[str(classifier)] = Series({'accuracy': score, 'time': time})

            df['ranks'] = self.weight * df['accuracy'].rank(ascending=False) + (1 - self.weight) * df['time'].rank()
            df = df.sort_values(by=['ranks'])

        print(f"Best performing algorithm: {df.index[0]}")

        return df


output = GridSearchClassifier(train_set, test_set, train_targets, test_targets, classifiers).fit()
