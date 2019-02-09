'''
Module for sklearn classifiers. Follows structure implemented by
scikit-learn; linear models, trees, etc.. one file per module


Import modules to register class names in global registry
'''
from . import base_sklearn_classifier
from . import dummy
from . import ensemble
from . import gaussian_process
from . import linear_model
from . import mixture
from . import multiclass
from . import multioutput
from . import naive_bayes
from . import neighbors
from . import neural_network
from . import svm
from . import tree


__author__ = 'Elisha Yadgaran'
