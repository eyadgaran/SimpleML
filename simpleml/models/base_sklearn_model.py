'''
Base module for Sklearn models.
'''

__author__ = 'Elisha Yadgaran'


import inspect
import logging

from simpleml.constants import TRAIN_SPLIT

from .base_model import LibraryModel

LOGGER = logging.getLogger(__name__)


class SklearnModel(LibraryModel):
    '''
    No different than base model. Here just to maintain the pattern
    Generic Base -> Library Base -> Domain Base -> Individual Models
    (ex: [Library]Model -> SklearnModel -> SklearnClassifier -> SklearnLogisticRegression)
    '''

    def _fit(self):
        '''
        Separate out actual fit call for optional overwrite in subclasses

        Sklearn estimators don't support data generators, so do not expose
        fit_generator method
        '''
        # Explicitly fit only on default (train) split
        split = self.transform(X=None, dataset_split=TRAIN_SPLIT, return_generator=False)
        supported_fit_params = {}

        # Ensure input compatibility with split object
        fit_params = inspect.signature(self.external_model.fit).parameters
        # check if any params are **kwargs (all inputs accepted)
        has_kwarg_params = any([param.kind == param.VAR_KEYWORD for param in fit_params.values()])
        # log ignored args
        if not has_kwarg_params:
            for split_arg, val in split.items():
                if split_arg not in fit_params:
                    LOGGER.warning(f'Unsupported fit param encountered, `{split_arg}`. Dropping...')
                else:
                    supported_fit_params[split_arg] = val
        else:
            supported_fit_params = split

        self.external_model.fit(**supported_fit_params)
