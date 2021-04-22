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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.dummy import DummyClassifier
from pandas import DataFrame, Series
from numpy import ndarray, float64, dtype
from tabulate import tabulate


def scikit_learn_classifiers():
    """
    Returns all appropriate scikit learn classifiers in a list.
    """
    classifiers = [DummyClassifier(strategy='most_frequent'), DecisionTreeClassifier(), KNeighborsClassifier(),
                   RandomForestClassifier(), MLPClassifier(), SVC(), AdaBoostClassifier(), GaussianProcessClassifier()]

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
    scaler: scikit-learn scaler, default=StandardScaler()
        Scaler from the Scikit-Learn library.
    score: str, (default='accuarcy_score')
        score used to measure performance options - 'accuracy_score', 'f1_score'
    """

    def __init__(self, train_set, test_set, train_targets, test_targets, classifiers, weights=None,
                 scaler=StandardScaler(), score='accuracy_score'):
        """
        Error logging performed and variables set.
        """

        if weights is None:
            weights = [Fraction(1, 3)] * 3

        variables = [train_set, test_set, train_targets, test_targets, classifiers, weights]
        variable_names = ['train_set', 'test_set', 'train_targets',
                          'test_targets', 'classifiers', 'weights', 'score']
        types = [ndarray, ndarray, ndarray, ndarray, list, list, str]

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

        if score not in ['accuracy_score', 'f1_score']:
            raise ValueError(f"{score} is not an option for score. "
                             f"The current options for score are 'accuracy_score', 'f1_score'")

        self.train_set = train_set
        self.test_set = test_set
        self.train_targets = train_targets
        self.test_targets = test_targets
        self.scaler = scaler
        self.classifiers = classifiers
        self.weights = weights
        self.score = score

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

        global output_score
        fitted_scaler = scaler.fit(self.train_set)
        train_set = fitted_scaler.transform(self.train_set)
        test_set = fitted_scaler.transform(self.test_set)

        t = time()
        fitted_model = model.fit(X=train_set, y=self.train_targets)
        train_time = time() - t

        t2 = time()
        predictions = fitted_model.predict(test_set)
        test_time = time() - t2

        if self.score == 'accuracy_score':
            output_score = accuracy_score(y_true=self.test_targets, y_pred=predictions)
        elif self.score == 'f1_score':
            output_score = f1_score(y_true=self.test_targets, y_pred=predictions)

        return output_score, train_time, test_time

    def fit(self):
        """
        Fit all classifiers in list to the train-test pipeline. Rank best performing on the basis of (weighted)
        accuracy and time.

        Returns
        --------
        df: pandas.DataFrame
            DataFrame ranking each algorithm in order. Algorithm name is the index. Accuracy, time and ranks are reported.
        """

        df = DataFrame(columns=['accuracy', 'train_time', 'test_time'])

        for classifier in self.classifiers:
            print(f'Testing algorithm: {classifier}')
            output_score, train_time, test_time = self._pipeline(scaler=self.scaler, model=classifier)

            df.loc[str(classifier)] = Series(
                {'accuracy': output_score, 'train_time': train_time, 'test_time': test_time})

            df['ranks'] = self.weights[0] * df['accuracy'].rank(ascending=False) + \
                          self.weights[1] * df['train_time'].rank() + \
                          self.weights[2] * df['test_time'].rank()

            df = df.sort_values(by=['ranks'])

        print(f"Best performing algorithm: {df.index[0]}")

        return df


class TestGridSearchClassifierOutput(TestCase):
    """
    Tests that monitor changes to GridSearchClassifier. These tests allow for changes to be made to the source code
    and to understand if outputs are effected.
    """

    def __init__(self, *args, **kwargs):
        super(TestGridSearchClassifierOutput, self).__init__(*args, **kwargs)

        train_set, train_targets = make_classification(n_samples=1000, n_features=4,
                                                       n_informative=2, n_redundant=0,
                                                       random_state=0, shuffle=False)

        test_set, test_targets = make_classification(n_samples=1000, n_features=4,
                                                     n_informative=2, n_redundant=0,
                                                     random_state=6, shuffle=False)

        classifiers = scikit_learn_classifiers()

        seed(123)

        self.output = GridSearchClassifier(train_set, test_set, train_targets,
                                           test_targets, classifiers, [1, 0, 0]).fit()

        print('Full output')
        print(list(self.output.columns))
        print(tabulate(self.output))

    def test_column_names(self):
        self.assertListEqual(list(self.output.columns),
                             ['accuracy', 'train_time', 'test_time', 'ranks'])

    def test_shape(self):
        self.assertTupleEqual(self.output.shape, (8, 4))

    def test_dtypes(self):
        for x in list(self.output.columns):
            try:
                self.assertEqual(self.output[x].dtype, float64)
            except AssertionError as e:
                raise AssertionError(f'{e}. Column: {x}')

    def test_min_of_columns(self):
        for x in list(self.output.columns):
            try:
                self.assertGreaterEqual(self.output[x].min(), 0)
            except AssertionError as e:
                raise AssertionError(f'{e}. Column: {x}')

    def test_max_accuracy(self):
        self.assertLessEqual(self.output['accuracy'].max(), 1)

    def test_index_dtype(self):
        self.assertEqual(self.output.index.dtype, dtype('O'))


main() if __name__ == '__main__' else None
