"""
GridSearchScikitLearn

Module to use a grid search to select an algorithm from scikit-learn for supervised learning tasks.

Author: G Bettsworth

2020
"""

from fractions import Fraction
from random import seed
from time import time
from unittest import TestCase, main

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
from numpy import ndarray, array

from itertools import product


def scikit_learn_classifiers():
    """
    Returns all appropriate scikit learn classifiers in a list.
    """
    classifiers = [DummyClassifier(strategy='most_frequent'), DecisionTreeClassifier(), KNeighborsClassifier(),
                   RandomForestClassifier(),
                   MLPClassifier(), SVC(), AdaBoostClassifier(), GaussianProcessClassifier()]

    return classifiers


def scikit_learn_classifiers_and_parameters():
    """
    Returns all appropriate scikit learn classifiers with all possible parameters in a list.
    """

    classifiers = []

    criterion = ['gini', 'entropy']
    splitter = ['best', 'random']
    max_features = ['auto', 'sqrt', 'log2', None]

    # the ideal min_samples_split values tend to be between 1 to 40 for the
    # CART algorithm which is the algorithm implemented in scikit-learn (Mantovani et al, 2018).
    min_samples_split = list(range(2, 41))

    dt_params = list(product(criterion, splitter, max_features, min_samples_split))

    for x in dt_params:
        classifiers.append(DecisionTreeClassifier(criterion=x[0],
                                                  splitter=x[1],
                                                  max_features=x[2],
                                                  min_samples_split=x[3]))

    n_estimators = list(range(2, 401))

    rf_params = list(product(n_estimators, criterion, max_features, min_samples_split))

    for x in rf_params:
        classifiers.append(RandomForestClassifier(n_estimators=x[0],
                                                  criterion=x[1],
                                                  max_features=x[2],
                                                  min_samples_split=x[3]))

    return classifiers


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
    weights: list of float, default=[Fraction(1, 3)]*3
        Weight for ranking the importance of accuracy and time as a measure of performance.
        The first element of the list is the weight for accuracy. The second element is the weight for training
        time. The third element is the weight for testing time (a proxy for implementation time).
        The weights must add up to 1.
        The default is weight each number equally.
    scaler: , default=StandardScaler()
        Scaler from the Scikit-Learn library.
    """

    def __init__(self, train_set, test_set, train_targets, test_targets, classifiers, weights=[Fraction(1, 3)] * 3,
                 scaler=StandardScaler()):
        """
        Error logging performed and variables set.
        """

        variables = [train_set, test_set, train_targets, test_targets, classifiers, weights]
        variable_names = ['train_set', 'test_set', 'train_targets', 'test_targets', 'classifiers', 'weights']
        types = [ndarray, ndarray, ndarray, ndarray, list, list]

        for variable, name, variable_type in zip(variables, variable_names, types):
            if not isinstance(variable, variable_type):
                raise TypeError(f"The variable {name} needs to be a {variable_type}")

        for weight in weights:
            if weight > 1.0:
                raise ValueError(f"You have specified weight={weight}. This weight is too large,"
                                 f" it must be less than or equal to 1 and greater than or equal to 0.")

            if weight < 0.0:
                raise ValueError(f"You have specified weight={weight}. This weight is too small,"
                                 f" it must be greater than or equal to 0 and less than or equal to 1.")

        if sum(weights) != 1:
            raise ValueError(f"The weights must add up to 1. The weights currently add up to {sum(weights)}")

        self.train_set = train_set
        self.test_set = test_set
        self.train_targets = train_targets
        self.test_targets = test_targets
        self.scaler = scaler
        self.classifiers = classifiers
        self.weights = weights

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

        fitted_scaler = scaler.fit(self.train_set)
        train_set = fitted_scaler.transform(self.train_set)
        test_set = fitted_scaler.transform(self.test_set)

        t = time()
        fitted_model = model.fit(X=train_set, y=self.train_targets)
        train_time = time() - t

        t2 = time()
        predictions = fitted_model.predict(test_set)
        test_time = time() - t2

        score = accuracy_score(y_true=self.test_targets, y_pred=predictions)

        return score, train_time, test_time

    def fit(self):
        """
        Fit all classifiers in list to the train-test pipeline. Rank best performing on the basis of (weighted)
        accuracy and time.

        Returns
        --------
        df: pandas.DataFrame
            DataFrame ranking each algorithm in order. Algorithm name is the index. Accuracy, time and ranks are reported.
        """

        classifier_strings = [str(x) for x in self.classifiers]

        df = DataFrame(columns=['accuracy', 'train_time', 'test_time'], index=classifier_strings)

        for classifier in self.classifiers:
            score, train_time, test_time = self._pipeline(scaler=self.scaler, model=classifier)

            try:
                df.loc[str(classifier)] = Series({'accuracy': score, 'train_time': train_time, 'test_time': test_time})
            except ValueError:
                df.loc[classifier] = Series({'accuracy': score, 'train_time': train_time, 'test_time': test_time})

            df['ranks'] = self.weights[0] * df['accuracy'].rank(ascending=False) + \
                          self.weights[1] * df['train_time'].rank() + \
                          self.weights[2] * df['test_time'].rank()

            df = df.sort_values(by=['ranks'])

        print(f"Best performing algorithm: {df.index[0]}")

        return df


# TODO - DecisionTree, AdaBoost and RandomForest all fit perfectly which means order changes.


class TestGridSearchClassifierOutputIsUnchanged(TestCase):
    """
    Tests that monitor changes to GridSearchClassifier. These tests allow for changes to be made to the source code
    and to understand if outputs are effected.
    """

    def __init__(self, *args, **kwargs):
        super(TestGridSearchClassifierOutputIsUnchanged, self).__init__(*args, **kwargs)

        train_set = array([[1, 2, 3], [4, 5, 6], [1, 2, 3], [7, 8, 9], [1, 2, 4]])
        train_targets = array([1, 0, 1, 0, 1])

        test_set = array([[1, 2, 3], [4, 5, 6], [1, 2, 4], [7, 9, 9]])
        test_targets = array([1, 0, 1, 0])

        classifiers = scikit_learn_classifiers()

        seed(123)

        # ranked on accuracy only to ensure tests are reproducible
        t = time()
        output = GridSearchClassifier(train_set, test_set, train_targets,
                                      test_targets, classifiers, [1, 0, 0]).fit()

        self.output = output.sort_index()
        self.overall_time = t - time()

    def tests_index_of_output(self):
        """
        Test that the index of the output dataframe is as expected.
        """
        expected = ['AdaBoostClassifier()',
                    'DecisionTreeClassifier()',
                    "DummyClassifier(strategy='most_frequent')",
                    'GaussianProcessClassifier()',
                    'KNeighborsClassifier()',
                    'MLPClassifier()',
                    'RandomForestClassifier()',
                    'SVC()']

        print("TEST INDEX OF OUTPUT")
        print("Actual index:")
        print(list(self.output.index))

        self.assertListEqual(list(self.output.index), expected)

    def test_accuracy_is_as_expected(self):
        """
        Test that the accuracy column is as expected.
        """

        expected = [1.0, 1.0, 0.5, 1.0, 0.5, 1.0, 1.0, 1.0]

        print("TEST ACCURACY COLUMN OF OUTPUT")
        print("Actual output:")
        print(list(self.output.accuracy))

        self.assertListEqual(list(self.output.accuracy), expected)


main() if __name__ == '__main__' else None
