"""
Author : G Bettsworth
"""

import numpy as np
import research_pipeline as rp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
    pca_fits[name] = pca_fit

# now there are the original versions of each one plus the pca versions

# implement research pipeline for all sets

accuracy_ratio, conf_matrix, precision_value, recall_value, f1 = rp.research_pipeline(train_set=arrays['a_yn'],
                                                                                      train_targets=targets['a_yn'],
                                                                                      test_set=arrays['b_yn'],
                                                                                      test_targets=targets['b_yn'],
                                                                                      knn=True,
                                                                                      svm=False,
                                                                                      kfolds=5)

print(accuracy_ratio, conf_matrix, precision_value, recall_value, f1)

accuracy_ratio, conf_matrix, precision_value, recall_value, f1 = rp.research_pipeline(train_set=arrays['a_yn'],
                                                                                      train_targets=targets['a_yn'],
                                                                                      test_set=arrays['b_yn'],
                                                                                      test_targets=targets['b_yn'],
                                                                                      knn=False,
                                                                                      svm=True,
                                                                                      kfolds=5)

print(accuracy_ratio, conf_matrix, precision_value, recall_value, f1)

