from simpleml import TRAIN_SPLIT
from simpleml.persistables.base_persistable import BasePersistable
from simpleml.persistables.saving import AllSaveMixin
from simpleml.pipelines.external_pipelines import DefaultPipeline, SklearnPipeline
from simpleml.utils.errors import PipelineError

from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import JSONB
import logging

__author__ = 'Elisha Yadgaran'


LOGGER = logging.getLogger(__name__)


class BasePipeline(BasePersistable, AllSaveMixin):
    '''
    Base class for all Pipelines objects.

    -------
    Schema
    -------
    params: pipeline parameter metadata for easy insight into hyperparameters across trainings
    '''
    __abstract__ = True

    # Additional pipeline specific metadata
    params = Column(JSONB, default={})

    def __init__(self, has_external_files=True, transformers=[],
                 **kwargs):
        external_pipeline_class = kwargs.pop('external_pipeline_class', 'default')

        super(BasePipeline, self).__init__(
            has_external_files=has_external_files, **kwargs)

        # Instantiate pipeline
        self.config['external_pipeline_class'] = external_pipeline_class
        self.object_type = 'PIPELINE'
        self._external_file = self._create_external_pipeline(
            external_pipeline_class, transformers, **kwargs)
        # Initialize as unfitted
        self.state['fitted'] = False

    @property
    def external_pipeline(self):
        '''
        All pipeline objects are going to require some filebase persisted object

        Wrapper around whatever underlying class is desired
        (eg sklearn or native)
        '''
        if self.unloaded_externals:
            self._load_external_files()

        return self._external_file

    def _create_external_pipeline(self, external_pipeline_class, transformers,
                                  **kwargs):
        '''
        should return the desired pipeline object

        :param external_pipeline_class: str of class to use, can be 'default' or 'sklearn'
        '''
        if external_pipeline_class == 'default':
            return DefaultPipeline(transformers)
        elif external_pipeline_class == 'sklearn':
            return SklearnPipeline(transformers, **kwargs)
        else:
            raise NotImplementedError('Only default or sklearn pipelines supported')

    def add_dataset(self, dataset):
        '''
        Setter method for dataset used
        '''
        self.dataset = dataset

    def add_transformer(self, name, transformer):
        '''
        Setter method for new transformer step
        '''
        self.external_pipeline.add_transformer(name, transformer)
        # Need to refit now
        self.state['fitted'] = False

    def remove_transformer(self, name):
        '''
        Delete method for transformer step
        '''
        self.external_pipeline.remove_transformer(name)
        # Need to refit now
        self.state['fitted'] = False

    def _hash(self):
        '''
        Hash is the combination of the:
            1) Dataset
            2) Transformers
            3) Transformer Params
            4) Pipeline Config
        '''
        dataset_hash = self.dataset.hash_ or self.dataset._hash()
        transformers = self.get_transformers()
        transformer_params = self.get_params()
        pipeline_config = self.config

        return self.custom_hasher((dataset_hash, transformers, transformer_params, pipeline_config))

    def save(self, **kwargs):
        '''
        Extend parent function with a few additional save routines

        1) save params
        2) save transformer metadata
        3) features
        '''
        if self.dataset is None:
            raise PipelineError('Must set dataset before saving')

        if not self.state['fitted']:
            raise PipelineError('Must fit pipeline before saving')

        self.params = self.get_params(**kwargs)
        self.metadata_['transformers'] = self.get_transformers()
        self.metadata_['feature_names'] = self.get_feature_names()

        super(BasePipeline, self).save(**kwargs)

        # Sqlalchemy updates relationship references after save so reload class
        self.dataset.load(load_externals=False)

    def load(self, **kwargs):
        '''
        Extend main load routine to load relationship class
        '''
        super(BasePipeline, self).load(**kwargs)

        # By default dont load data unless it actually gets used
        self.dataset.load(load_externals=False)

    def get_dataset_split(self, split=None):
        '''
        Get specific dataset split
        By default no constraint imposed, but convention is that return should
        be a tuple of (X, y)
        '''
        if split is None:
            split = TRAIN_SPLIT

        if not hasattr(self, '_dataset_splits') or self._dataset_splits is None:
            self.split_dataset()

        return self._dataset_splits.get(split)

    def fit(self, **kwargs):
        '''
        Pass through method to external pipeline
        '''
        if self.dataset is None:
            raise PipelineError('Must set dataset before fitting')

        if self.state['fitted']:
            LOGGER.warning('Cannot refit pipeline, skipping operation')
            return self

        # Only use train fold to fit
        X, y = self.get_dataset_split(TRAIN_SPLIT)
        self.external_pipeline.fit(X, y, **kwargs)
        self.state['fitted'] = True

        return self

    def transform(self, X, dataset_split=None, return_y=False, **kwargs):
        '''
        Pass through method to external pipeline

        :param X: dataframe/matrix to transform, if None, use internal dataset
        :param return_y: whether to return y with output - only used if X is None
            necessary for fitting a supervised model after
        '''
        if not self.state['fitted']:
            raise PipelineError('Must fit pipeline before transforming')

        if X is None:
            X, y = self.get_dataset_split(dataset_split)
            output = self.external_pipeline.transform(X, **kwargs)

            if return_y:
                return output, y

            return output

        return self.external_pipeline.transform(X, **kwargs)

    def fit_transform(self, return_y=False, **kwargs):
        '''
        Wrapper for fit and transform methods
        ASSUMES only applies to train split

        :param return_y: whether to return y with output
            necessary for fitting a supervised model after
        '''
        self.fit(**kwargs)
        output, y = self.transform(X=None, dataset_split=TRAIN_SPLIT, return_y=True, **kwargs)

        if return_y:
            return output, y

        return output

    def get_params(self, **kwargs):
        '''
        Pass through method to external pipeline
        '''
        return self.external_pipeline.get_params(**kwargs)

    def set_params(self, **params):
        '''
        Pass through method to external pipeline
        '''
        return self.external_pipeline.set_params(**params)

    def get_transformers(self):
        '''
        Pass through method to external pipeline
        '''
        return self.external_pipeline.get_transformers()

    def get_feature_names(self):
        '''
        Pass through method to external pipeline
        Should return a list of the final features generated by this pipeline
        '''
        initial_features = self.dataset.get_feature_names()
        return self.external_pipeline.get_feature_names(feature_names=initial_features)
