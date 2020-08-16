"""
Author : G Bettsworth
"""

import numpy as np
import random
import research_pipeline as rp

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

data_location = r"../binary_classifiers_comparison_data"
output_location = r"../binary_classifiers_comparison_outputs"

random.seed(123)

array_a = np.genfromtxt(rf"{data_location}/a_wh_question_datapoints.txt", skip_header=1)
target_a = np.genfromtxt(rf"{data_location}/a_wh_question_targets.txt")

array_b = np.genfromtxt(rf"{data_location}/b_wh_question_datapoints.txt", skip_header=1)
target_b = np.genfromtxt(rf"{data_location}/b_wh_question_targets.txt")

knn_outputs = rp.sklearn_prediction_pipeline(train_set=array_a,
                                            train_targets=target_a,
                                            test_set=array_b,
                                            test_targets=target_b,
                                            model=KNeighborsClassifier())

svc_outputs = rp.sklearn_prediction_pipeline(train_set=array_a,
                                            train_targets=target_a,
                                            test_set=array_b,
                                            test_targets=target_b,
                                            model=SVC())

rf_outputs = rp.sklearn_prediction_pipeline(train_set=array_a,
                                            train_targets=target_a,
                                            test_set=array_b,
                                            test_targets=target_b,
                                            model=RandomForestClassifier())

gb_outputs = rp.sklearn_prediction_pipeline(train_set=array_a,
                                            train_targets=target_a,
                                            test_set=array_b,
                                            test_targets=target_b,
                                            model=GradientBoostingClassifier())

extree_outputs = rp.sklearn_prediction_pipeline(train_set=array_a,
                                            train_targets=target_a,
                                            test_set=array_b,
                                            test_targets=target_b,
                                            model=ExtraTreesClassifier())

dt_outputs = rp.sklearn_prediction_pipeline(train_set=array_a,
                                            train_targets=target_a,
                                            test_set=array_b,
                                            test_targets=target_b,
                                            model=DecisionTreeClassifier())

tf_accuracy = rp.tenserflow_prediction_pipeline(train_set=array_a,
                                        train_targets=target_a,
                                        test_set=array_b,
                                        test_targets=target_b)