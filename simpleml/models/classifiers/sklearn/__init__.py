'''
Module for sklearn classifiers. Follows structure implemented by
scikit-learn; linear models, trees, etc.. one file per module


Import modules to register class names in global registry
'''
from . import (
    base_sklearn_classifier,
    dummy,
    ensemble,
    gaussian_process,
    linear_model,
    mixture,
    multiclass,
    multioutput,
    naive_bayes,
    neighbors,
    neural_network,
    svm,
    tree,
)

__author__ = 'Elisha Yadgaran'
