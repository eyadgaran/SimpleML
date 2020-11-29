'''
Base module for Sklearn models.
'''

__author__ = 'Elisha Yadgaran'


from simpleml.constants import TRAIN_SPLIT
from .base_model import LibraryModel


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
        self.external_model.fit(**split)
