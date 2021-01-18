"""
Example of GridSearchClassifier on face recognition.

The task is to correctly identify when someone was a question beginning with 'wh' (so when/where).

Data has been labelled.
"""

from GridSearchScikitLearn import GridSearchClassifier, scikit_learn_classifiers

from numpy import genfromtxt

array_a = genfromtxt("./a_wh_question_datapoints.txt", skip_header=1)
target_a = genfromtxt("./a_wh_question_targets.txt")

array_b = genfromtxt("./b_wh_question_datapoints.txt", skip_header=1)
target_b = genfromtxt("./b_wh_question_targets.txt")

output = GridSearchClassifier(train_set=array_a,
                              test_set=array_b,
                              train_targets=target_a,
                              test_targets=target_b,
                              classifiers=scikit_learn_classifiers()).fit()

output = GridSearchClassifier(train_set=array_b,
                              test_set=array_a,
                              train_targets=target_b,
                              test_targets=target_a,
                              classifiers=scikit_learn_classifiers()).fit()
