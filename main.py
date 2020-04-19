"""
Author : G Bettsworth
"""

import numpy as np
import pandas as pd
import time
import itertools
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import research_pipeline as rp

data_location = r"../binary_classifiers_comparison_data"
output_location = r"../binary_classifiers_comparison_outputs"

array_dictionary_keys = ['a_wh', 'a_yn', 'b_wh', 'b_yn']
arrays = {}
targets = {}

### data prep

for name in array_dictionary_keys:
    array = np.genfromtxt(rf"{data_location}/{name}_question_datapoints.txt", skip_header=1)
    arrays[name] = array

    target = np.genfromtxt(rf"{data_location}/{name}_question_targets.txt")
    targets[name] = target

# principle components analysis to create another feature representation
# how do you choose number of components?
pca = PCA(n_components=10)

pca_fits = {}

for name in array_dictionary_keys:
    scaled = StandardScaler().fit_transform(arrays[name])
    pca_fit = pca.fit_transform(scaled)
    pca_fits[name + '_pca'] = pca_fit

arrays.update(pca_fits)

# now there are the original versions of each one plus the pca versions

# implement research pipeline for all sets

performances = pd.DataFrame()

trains = ['a_wh', 'a_yn', 'b_wh', 'b_yn', 'a_wh_pca', 'a_yn_pca', 'b_wh_pca', 'b_yn_pca'] * 2

train_targets = ['a_wh', 'a_yn', 'b_wh', 'b_yn'] * 4

tests = ['b_wh', 'b_yn','a_wh', 'a_yn', 'b_wh_pca', 'b_yn_pca', 'a_wh_pca', 'a_yn_pca'] * 2

test_targets = ['b_wh', 'b_yn', 'a_wh', 'a_yn'] * 4

models = ["knn"] * 8 + ["svm"] * 8

for (train_set, train_targets, test_set, test_targets, model) in zip(trains, train_targets, tests, test_targets, models):

    t = time.time()

    print(f"pipeline: train: {train_set}, test: {test_set}, using {model}")

    accuracy_ratio, conf_matrix, precision_value, recall_value, f1 = rp.research_pipeline(train_set=arrays[train_set],
                                                                                          train_targets=targets[train_targets],
                                                                                          test_set=arrays[test_set],
                                                                                          test_targets=targets[test_targets],
                                                                                          model=model,
                                                                                          kfolds=5)

    t2 = time.time() - t

    other = pd.DataFrame([train_set, train_targets, model, accuracy_ratio,
                          conf_matrix, precision_value, recall_value, f1, t2])

    other = other.transpose()

    performances = pd.concat([other, performances], ignore_index=True, axis=0)


performances.columns = ['train_set', 'test_set', 'model', 'accuracy_ratio',
                        'confusion_matrix', 'precision', 'recall', 'f1', 'time']

performances.to_csv(r"../binary_classifiers_comparison_outputs/model_performances.csv", index=False)
