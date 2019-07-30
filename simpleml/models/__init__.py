'''
Import modules to register class names in global registry

Expose classes in one import module

Models Directory follows this structure:
Generic Base -> Library Base -> Domain Base -> Individual Models
(ex: Model -> SklearnModel -> SklearnClassifier -> SklearnLogisticRegression)
'''

__author__ = 'Elisha Yadgaran'


# Exposed Classes
# Base Classes
from .base_model import Model, LibraryModel
from .base_sklearn_model import SklearnModel
from .base_keras_model import KerasModel

# Sklearn Classifiers
from .classifiers.sklearn.base_sklearn_classifier import SklearnClassifier
from .classifiers.sklearn.dummy import SklearnDummyClassifier
from .classifiers.sklearn.ensemble import SklearnAdaBoostClassifier, SklearnBaggingClassifier,\
    SklearnExtraTreesClassifier, SklearnGradientBoostingClassifier, SklearnRandomForestClassifier,\
    SklearnVotingClassifier
from .classifiers.sklearn.gaussian_process import SklearnGaussianProcessClassifier
from .classifiers.sklearn.linear_model import SklearnLogisticRegression, SklearnLogisticRegressionCV,\
    SklearnPerceptron, SklearnRidgeClassifier, SklearnRidgeClassifierCV, SklearnSGDClassifier
from .classifiers.sklearn.mixture import SklearnBayesianGaussianMixture, SklearnGaussianMixture
from .classifiers.sklearn.multiclass import SklearnOneVsRestClassifier, SklearnOneVsOneClassifier,\
    SklearnOutputCodeClassifier
from .classifiers.sklearn.multioutput import SklearnClassifierChain, SklearnMultiOutputClassifier
from .classifiers.sklearn.naive_bayes import SklearnBernoulliNB, SklearnGaussianNB, SklearnMultinomialNB
from .classifiers.sklearn.neighbors import SklearnKNeighborsClassifier
from .classifiers.sklearn.neural_network import SklearnMLPClassifier
from .classifiers.sklearn.svm import SklearnLinearSVC, SklearnNuSVC, SklearnSVC
from .classifiers.sklearn.tree import SklearnDecisionTreeClassifier, SklearnExtraTreeClassifier

# Keras Classifiers
from .classifiers.keras.base_keras_classifier import KerasClassifier
from .classifiers.keras.model import KerasModelClassifier
from .classifiers.keras.seq2seq import KerasSeq2SeqClassifier, KerasEncoderDecoderClassifier,\
    KerasEncoderDecoderStateClassifier, KerasEncoderDecoderStatelessClassifier
from .classifiers.keras.sequential import KerasSequentialClassifier


# Internal Modules only - Not importable from here, but still registered
# Assumes directories manage their own imports
# Classifiers
from . import classifiers

# Clusterers
from . import clusterers

# Regressors
from . import regressors

# Transfer Learning
from . import transfer
