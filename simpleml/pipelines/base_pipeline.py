'''
Base Module for Pipelines
'''

__author__ = 'Elisha Yadgaran'


from simpleml import TRAIN_SPLIT
from simpleml.persistables.base_persistable import Persistable
from simpleml.persistables.saving import AllSaveMixin
from simpleml.persistables.meta_registry import PipelineRegistry
from simpleml.persistables.guid import GUID

from simpleml.pipelines.external_pipelines import DefaultPipeline, SklearnPipeline
from simpleml.pipelines.validation_split_mixins import Split
from simpleml.utils.errors import PipelineError

from sqlalchemy import Column, ForeignKey, UniqueConstraint, Index
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB
from future.utils import with_metaclass
import logging
import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


class AbstractPipeline(with_metaclass(PipelineRegistry, Persistable, AllSaveMixin)):
    '''
    Abstract Base class for all Pipelines objects.

    Relies on mixin classes to define the split_dataset method. Will throw
    an error on use otherwise

    -------
    Schema
    -------
    params: pipeline parameter metadata for easy insight into hyperparameters across trainings
    '''
    __abstract__ = True

    # Additional pipeline specific metadata
    params = Column(JSONB, default={})

    object_type = 'PIPELINE'

    def __init__(self, has_external_files=True, transformers=[],
                 external_pipeline_class='default', fitted=False,
                 **kwargs):
        super(AbstractPipeline, self).__init__(
            has_external_files=has_external_files, **kwargs)

        # Instantiate pipeline
        self.config['external_pipeline_class'] = external_pipeline_class
        self._external_file = self._create_external_pipeline(
            external_pipeline_class, transformers, **kwargs)
        # Initialize fit state -- pass as true to skip fitting transformers
        self.fitted = fitted

    @property
    def fitted(self):
        return self.state.get('fitted')

    @fitted.setter
    def fitted(self, value):
        self.state['fitted'] = value

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

    def assert_dataset(self, msg=''):
        '''
        Helper method to raise an error if dataset isn't present
        '''
        if self.dataset is None:
            raise PipelineError(msg)

    def assert_fitted(self, msg=''):
        '''
        Helper method to raise an error if pipeline isn't fit
        '''
        if not self.fitted:
            raise PipelineError(msg)

    def add_transformer(self, name, transformer):
        '''
        Setter method for new transformer step
        '''
        self.external_pipeline.add_transformer(name, transformer)
        # Need to refit now
        self.fitted = False

    def remove_transformer(self, name):
        '''
        Delete method for transformer step
        '''
        self.external_pipeline.remove_transformer(name)
        # Need to refit now
        self.fitted = False

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
        self.assert_dataset('Must set dataset before saving')
        self.assert_fitted('Must fit pipeline before saving')

        self.params = self.get_params(**kwargs)
        self.metadata_['transformers'] = self.get_transformers()
        self.metadata_['feature_names'] = self.get_feature_names()

        # Skip file-based persistence if there are no transformers
        if not self.get_transformers():
            self.has_external_files = False

        super(AbstractPipeline, self).save(**kwargs)

        # Sqlalchemy updates relationship references after save so reload class
        self.dataset.load(load_externals=False)

    def load(self, **kwargs):
        '''
        Extend main load routine to load relationship class
        '''
        super(AbstractPipeline, self).load(**kwargs)

        # Create dummy pipeline if one wasnt saved
        if not self.has_external_files:
            self._external_file = self._create_external_pipeline(
                self.config['external_pipeline_class'], [], **self.params)

        # By default dont load data unless it actually gets used
        self.dataset.load(load_externals=False)

    def get_dataset_split(self, split=None):
        '''
        Get specific dataset split
        Assumes a Split object (`simpleml.pipelines.validation_split_mixins.Split`)
        is returned. Inherit or implement similar expected attributes to replace

        Uses internal `self._dataset_splits` as the split container - assumes
        dictionary like itemgetter
        '''
        if split is None:
            split = TRAIN_SPLIT

        if not hasattr(self, '_dataset_splits') or self._dataset_splits is None:
            self.split_dataset()

        return self._dataset_splits[split]

    def X(self, split=None):
        '''
        Get X for specific dataset split
        '''
        return self.get_dataset_split(split).X

    def y(self, split=None):
        '''
        Get labels for specific dataset split
        '''
        return self.get_dataset_split(split).y

    def fit(self, **kwargs):
        '''
        Pass through method to external pipeline
        '''
        self.assert_dataset('Must set dataset before fitting')

        if self.fitted:
            LOGGER.warning('Cannot refit pipeline, skipping operation')
            return self

        # Only use default (train) fold to fit
        # No constraint on split -- can be a dataframe, ndarray, or generator
        # but must be encased in a Split object
        split = self.get_dataset_split()

        # Hack for python <3.5 -- cant use fit(**split, **kwargs)
        temp_kwargs = kwargs.copy()
        temp_kwargs.update(split)

        self.external_pipeline.fit(**temp_kwargs)
        self.fitted = True

        return self

    def transform(self, X, dataset_split=None, return_y=False, **kwargs):
        '''
        Pass through method to external pipeline

        :param X: dataframe/matrix to transform, if None, use internal dataset
        :param return_y: whether to return y with output - only used if X is None
            necessary for fitting a supervised model after
        '''
        self.assert_fitted('Must fit pipeline before transforming')

        if X is None:  # Retrieve dataset split
            split = self.get_dataset_split(dataset_split)
            if split.X is None or (isinstance(split.X, pd.DataFrame) and split.X.empty):
                output = None  # Skip transformations on empty dataset
            else:
                output = self.external_pipeline.transform(split.X, **kwargs)

            if return_y:
                return output, split.y

            return output

        return self.external_pipeline.transform(X, **kwargs)

    def fit_transform(self, return_y=False, **kwargs):
        '''
        Wrapper for fit and transform methods
        ASSUMES only applies to default (train) split

        :param return_y: whether to return y with output
            necessary for fitting a supervised model after
        '''
        self.fit(**kwargs)
        return self.transform(X=None, return_y=return_y, **kwargs)

    '''
    Pass-through methods to external pipeline
    '''
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


class Pipeline(AbstractPipeline):
    '''
    Base class for all Pipeline objects.

    -------
    Schema
    -------
    dataset_id: foreign key relation to the dataset used as input
    '''
    __tablename__ = 'pipelines'

    dataset_id = Column(GUID, ForeignKey("datasets.id", name="pipelines_dataset_id_fkey"))
    dataset = relationship("Dataset", enable_typechecks=False, foreign_keys=[dataset_id])

    __table_args__ = (
        # Unique constraint for versioning
        UniqueConstraint('name', 'version', name='pipeline_name_version_unique'),
        # Index for searching through friendly names
        Index('pipeline_name_index', 'name'),
     )


class GeneratorPipeline(Pipeline):
    '''
    Generator form of pipeline. Overwrites standard methods with ones that
    return generator objects
    '''
    def get_dataset_split(self, split=None, infinite_loop=False, batch_size=32, shuffle=True, **kwargs):
        '''
        Get specific dataset split
        '''
        split = super(GeneratorPipeline, self).get_dataset_split(split)
        X = split.X
        y = split.y

        dataset_size = X.shape[0]
        if dataset_size == 0:  # Return None
            return

        # Extract indices to subsample from
        if isinstance(X, pd.DataFrame):
            indices = X.index.tolist()
        elif isinstance(X, np.ndarray):
            indices = np.arange(X.shape[0])
        else:
            raise NotImplementedError

        # Loop through and sample indefinitely
        first_run = True
        current_index = 0
        while True:
            if current_index == 0 and shuffle and not first_run:
                np.random.shuffle(indices)

            batch = indices[current_index:min(current_index + batch_size, dataset_size)]

            if y is not None and (isinstance(y, (pd.DataFrame, pd.Series)) and not y.empty):  # Supervised
                if isinstance(X, (pd.DataFrame, pd.Series)):
                    yield Split(X=X.loc[batch], y=np.stack(y.loc[batch].squeeze().values))
                else:
                    yield Split(X=X[batch], y=y[batch])
            else:  # Unsupervised
                if isinstance(X, (pd.DataFrame, pd.Series)):
                    yield Split(X=X.loc[batch])
                else:
                    yield Split(X=X[batch])

            current_index += batch_size

            # Loop so that infinite batches can be generated
            if current_index >= dataset_size:
                if infinite_loop:
                    current_index = 0
                    first_run = False
                else:
                    break

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
            generator_split = self.get_dataset_split(dataset_split, **kwargs)
            for batch in generator_split:
                output = self.external_pipeline.transform(batch.X, **kwargs)

                if return_y:
                    yield output, batch.y
                else:
                    yield output
        else:
            yield self.external_pipeline.transform(X, **kwargs)
