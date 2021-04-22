"""
Example of GridSearchClassifier on face recognition.

The task is to correctly identify when someone was asking a question beginning with 'wh' (so when/where).

Data has been labelled.
"""
from itertools import product
from sklearn.tree import DecisionTreeClassifier
from numpy import genfromtxt
from tabulate import tabulate
from grid_search_classifier import GridSearchClassifier, scikit_learn_classifiers

array_a = genfromtxt("data_a_wh_question_datapoints.txt", skip_header=1)
target_a = genfromtxt("data_a_wh_question_targets.txt")

array_b = genfromtxt("data_b_wh_question_datapoints.txt", skip_header=1)
target_b = genfromtxt("data_b_wh_question_targets.txt")

output_a_models = GridSearchClassifier(train_set=array_a,
                                       test_set=array_b,
                                       train_targets=target_a,
                                       test_targets=target_b,
                                       classifiers=scikit_learn_classifiers(),
                                       weights=[0.9, 0.05, 0.05]).fit()

print(tabulate(output_a_models, headers=list(output_a_models.columns)))

output_b_models = GridSearchClassifier(train_set=array_b,
                                       test_set=array_a,
                                       train_targets=target_b,
                                       test_targets=target_a,
                                       classifiers=scikit_learn_classifiers(),
                                       weights=[0.9, 0.05, 0.05]).fit()

print(tabulate(output_b_models, headers=list(output_b_models.columns)))

# Example looking at different decision tree options

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

output_a_dt_models = GridSearchClassifier(train_set=array_b,
                                          test_set=array_a,
                                          train_targets=target_b,
                                          test_targets=target_a,
                                          classifiers=classifiers).fit()

# use F1 instead
output_a_dt_models_f1 = GridSearchClassifier(train_set=array_b,
                                             test_set=array_a,
                                             train_targets=target_b,
                                             test_targets=target_a,
                                             classifiers=classifiers,
                                             score='f1_score').fit()
