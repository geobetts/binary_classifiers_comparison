from random import seed
from unittest import TestCase, main

from numpy import dtype, float64
from sklearn.datasets import make_classification
from tabulate import tabulate

from grid_search_classifier import GridSearchClassifier, scikit_learn_classifiers


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
