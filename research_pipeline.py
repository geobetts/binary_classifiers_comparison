"""
Author: G Bettsworth
"""

import numpy as np
import random
import time
import sklearn.metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
import tensorflow as tf
from tensorflow import keras


def sklearn_prediction_pipeline(train_set,
                            train_targets,
                            test_set,
                            test_targets,
                            model):
        
    t = time.time()

    train_set = StandardScaler().fit_transform(train_set)
    test_set = StandardScaler().fit_transform(test_set)
    
    
    performances = cross_validate(estimator=model.fit(train_set, train_targets), 
                                  X=train_set, 
                                  y=train_targets, 
                                  return_estimator=True)

    random.seed(123)

    model_selected = random.choice(performances['estimator'])

    neighbours = model_selected

    knn_fit = neighbours.fit(train_set, train_targets)

    predictions = knn_fit.predict(test_set)

    predictions = predictions.reshape(test_targets.shape)
    
    accuracy_ratio = sklearn.metrics.accuracy_score(y_true=test_targets, y_pred=predictions)

    conf_matrix = sklearn.metrics.confusion_matrix(y_true=test_targets, y_pred=predictions, labels=[1, 0])
    precision_value = sklearn.metrics.precision_score(y_true=test_targets, y_pred=predictions)
    recall_value = sklearn.metrics.recall_score(y_true=test_targets, y_pred=predictions)
    f1 = sklearn.metrics.f1_score(y_true=test_targets, y_pred=predictions)

    t2 = time.time() - t
    
    outputs = {'model' : str(model),
               'cross_validation': str(performances),
               'accuracy' : accuracy_ratio,
               'conf_matrix' : conf_matrix,
               'precision_value' : precision_value,
               'recall_value' : recall_value,
               'f1' : f1,
               't2' : t2}

    return outputs



def tenserflow_prediction_pipeline(train_set,
                                train_targets,
                                test_set,
                                test_targets):

    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.Dense(10)
    ])
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    
    model.fit(train_set, train_targets, epochs=10)
    
    test_loss, test_acc = model.evaluate(test_set, test_targets, verbose=2)
    
    return test_acc




