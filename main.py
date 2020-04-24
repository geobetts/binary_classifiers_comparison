"""
Author : G Bettsworth
"""

import numpy as np
import pandas as pd
import time
import itertools
from sklearn.manifold import TSNE
import csv
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import research_pipeline as rp
from sklearn.decomposition import FastICA

data_location = r"../binary_classifiers_comparison_data"
output_location = r"../binary_classifiers_comparison_outputs"

file_suffix = "all_ICA_5"
all_versions = True

random.seed(123)

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

fits = {}
info = {}

for name in array_dictionary_keys:
    print(f"{name} fit")
    components = 5
    model = FastICA(n_components=components, whiten=True, max_iter=10000)
    scaled = StandardScaler().fit_transform(arrays[name])
    fit = model.fit_transform(scaled)
    fits[name + '_fit'] = fit

arrays.update(fits)

# now there are the original versions of each one plus the pca versions

# implement research pipeline for all sets

performances = pd.DataFrame()

if all_versions:
    trains = ['a_wh', 'a_yn', 'b_wh', 'b_yn', 'a_wh_fit', 'a_yn_fit', 'b_wh_fit', 'b_yn_fit'] * 2

    train_targets = ['a_wh', 'a_yn', 'b_wh', 'b_yn'] * 4

    tests = ['b_wh', 'b_yn', 'a_wh', 'a_yn', 'b_wh_fit', 'b_yn_fit', 'a_wh_fit', 'a_yn_fit'] * 2

    test_targets = ['b_wh', 'b_yn', 'a_wh', 'a_yn'] * 4

    models = ["knn"] * 8 + ["svm"] * 8

if not all_versions:
    trains = ['a_wh_fit', 'a_yn_fit', 'b_wh_fit', 'b_yn_fit'] * 2

    train_targets = ['a_wh', 'a_yn', 'b_wh', 'b_yn'] * 2

    tests = ['b_wh_fit', 'b_yn_fit', 'a_wh_fit', 'a_yn_fit'] * 2

    test_targets = ['b_wh', 'b_yn', 'a_wh', 'a_yn'] * 2

    models = ["knn"] * 4 + ["svm"] * 4

for (train_set, train_targets, test_set, test_targets, model) in zip(trains, train_targets, tests, test_targets,
                                                                     models):
    print(f"pipeline: train: {train_set}, test: {test_set}, using {model}")

    outputs = rp.research_pipeline(train_set=arrays[train_set],
                                   train_set_name=train_set,
                                   train_targets=targets[train_targets],
                                   test_set=arrays[test_set],
                                   test_set_name=test_set,
                                   test_targets=targets[test_targets],
                                   model=model,
                                   kfolds=10)

    performances = pd.concat([outputs, performances], ignore_index=True, axis=0)

performances.columns = ['train_set', 'test_set', 'model', 'best_performances', 'chosen_performance', 'cv_accuracy_best',
                        'accuracy_ratio', 'confusion_matrix', 'precision', 'recall', 'f1', 'time']

performances.to_csv(rf"../binary_classifiers_comparison_outputs/model_performances_{file_suffix}.csv", index=False)
