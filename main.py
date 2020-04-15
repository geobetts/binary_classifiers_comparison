"""
Author : G Bettsworth
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data_location = r"../binary_classifiers_comparison_data"
output_location = r"../binary_classifiers_comparison_outputs"

# read in all data
array_dictionary_keys = ['a_wh', 'a_yn', 'b_wh', 'b_yn']
arrays = {}
targets = {}

for name in array_dictionary_keys:
    array = np.genfromtxt(rf"{data_location}/{name}_question_datapoints.txt", skip_header=1)
    arrays[name] = array

    target = np.genfromtxt(rf"{data_location}/{name}_question_targets.txt", skip_header=1)
    targets[name] = target

# principle components analysis
pca = PCA(n_components=10)

pca_fits = {}

for name in array_dictionary_keys:
    scaled = StandardScaler().fit_transform(arrays[name])
    pca_fit = pca.fit_transform(scaled)
    pca_fits[name] = pca_fit

# you get two components not a component for every line
x=1