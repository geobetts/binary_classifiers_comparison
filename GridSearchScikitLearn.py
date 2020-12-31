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

from sklearn.datasets import make_classification
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
from numpy import ndarray


def scikit_learn_classifiers():
    """
    Returns all appropriate scikit learn classifiers in a list.
    """
    classifiers = [DummyClassifier(strategy='most_frequent'), DecisionTreeClassifier(), KNeighborsClassifier(),
                   RandomForestClassifier(),
                   MLPClassifier(), SVC(), AdaBoostClassifier(), GaussianProcessClassifier()]

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
            print("--------------------------------------")
            print(f'Testing: {classifier}')
            score, train_time, test_time = self._pipeline(scaler=self.scaler, model=classifier)

            print(f'Accuracy: {score}')
            print(f'Training Time: {train_time}')
            print(f'Testing Time: {test_time}')
            print("--------------------------------------")

            df.loc[str(classifier)] = Series({'accuracy': score, 'train_time': train_time, 'test_time': test_time})

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

        train_set, train_targets = make_classification(n_samples=1000, n_features=4,
                                                       n_informative=2, n_redundant=0,
                                                       random_state=123, shuffle=False)

        test_set, test_targets = make_classification(n_samples=1000, n_features=4,
                                                     n_informative=2, n_redundant=0,
                                                     random_state=123, shuffle=False)

        classifiers = scikit_learn_classifiers()

        seed(123)

        # ranked on accuracy only to ensure tests are reproducible
        t = time()
        self.output = GridSearchClassifier(train_set, test_set, train_targets,
                                           test_targets, classifiers, [1, 0, 0]).fit()
        self.overall_time = t - time()

    def tests_index_of_output(self):
        """
        Test that the index of the output dataframe is as expected.
        """
        expected = ['DecisionTreeClassifier()', 'RandomForestClassifier()', 'AdaBoostClassifier()',
                    'MLPClassifier()', 'SVC()', 'GaussianProcessClassifier()', 'KNeighborsClassifier()',
                    "DummyClassifier(strategy='most_frequent')"]

        print("TEST INDEX OF OUTPUT")
        print("Actual index:")
        print(list(self.output.index))

        self.assertListEqual(list(self.output.index), expected)

    def test_accuracy_is_as_expected(self):
        """
        Test that the accuracy column is as expected.
        """

        expected = [1.0, 1.0, 1.0, 0.996, 0.996, 0.995, 0.993, 0.501]

        print("TEST ACCURACY COLUMN OF OUTPUT")
        print("Actual output:")
        print(list(self.output.accuracy))

        self.assertListEqual(list(self.output.accuracy), expected)


main() if __name__ == '__main__' else None
