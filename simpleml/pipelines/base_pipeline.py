'''
Base Module for Pipelines
'''

__author__ = 'Elisha Yadgaran'

import inspect
import logging
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, Union

from future.utils import with_metaclass

import pandas as pd
from simpleml.constants import TRAIN_SPLIT
from simpleml.datasets.dataset_splits import Split, SplitContainer
from simpleml.persistables.base_persistable import Persistable
from simpleml.persistables.sqlalchemy_types import GUID, MutableJSON
from simpleml.registries import PipelineRegistry
from simpleml.save_patterns.decorators import ExternalArtifactDecorators
from simpleml.utils.errors import PipelineError
from sqlalchemy import Column, ForeignKey, Index, UniqueConstraint
from sqlalchemy.orm import relationship

from .projected_splits import (IdentityProjectedDatasetSplit,
                               ProjectedDatasetSplit)

if TYPE_CHECKING:
    # Cyclical import hack for type hints
    from simpleml.datasets.base_dataset import Dataset


LOGGER = logging.getLogger(__name__)


@ExternalArtifactDecorators.register_artifact(
    artifact_name='pipeline', save_attribute='external_pipeline', restore_attribute='_external_file')
class AbstractPipeline(with_metaclass(PipelineRegistry, Persistable)):
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
    params = Column(MutableJSON, default={})

    object_type: str = 'PIPELINE'

    def __init__(self,
                 has_external_files: bool = True,
                 transformers: Optional[List[Any]] = None,
                 fitted: bool = False,
                 **kwargs):
        # If no save patterns are set, specify a default for disk_pickled
        if 'save_patterns' not in kwargs:
            kwargs['save_patterns'] = {'pipeline': ['disk_pickled']}
        super(AbstractPipeline, self).__init__(
            has_external_files=has_external_files, **kwargs)

        # Instantiate pipeline
        if transformers is None:
            transformers: List[Any] = []
        self._external_file = self._create_external_pipeline(transformers, **kwargs)
        # Initialize fit state -- pass as true to skip fitting transformers
        self.fitted = fitted

    '''
    Persistable Management
    '''

    @property
    def fitted(self) -> bool:
        return self.state.get('fitted')

    @fitted.setter
    def fitted(self, value: bool) -> None:
        self.state['fitted'] = value

    @property
    def external_pipeline(self) -> Any:
        '''
        All pipeline objects are going to require some filebase persisted object

        Wrapper around whatever underlying class is desired
        (eg sklearn or native)
        '''
        self.load_if_unloaded('pipeline')
        return self._external_file

    def _create_external_pipeline(self, *args, **kwargs):
        '''
        each subclass should instantiate the respective pipeline library
        '''
        raise NotImplementedError('Must use a specific pipeline implementation')

    def add_dataset(self, dataset: 'Dataset') -> None:
        '''
        Setter method for dataset used
        '''
        self.dataset = dataset

    def assert_dataset(self, msg: str = '') -> None:
        '''
        Helper method to raise an error if dataset isn't present
        '''
        if self.dataset is None:
            raise PipelineError(msg)

    def assert_fitted(self, msg: str = '') -> None:
        '''
        Helper method to raise an error if pipeline isn't fit
        '''
        if not self.fitted:
            raise PipelineError(msg)

    def add_transformer(self, name: str, transformer: Any) -> None:
        '''
        Setter method for new transformer step
        '''
        self.external_pipeline.add_transformer(name, transformer)
        # Need to refit now
        self.fitted = False

    def remove_transformer(self, name: str) -> None:
        '''
        Delete method for transformer step
        '''
        self.external_pipeline.remove_transformer(name)
        # Need to refit now
        self.fitted = False

    def _hash(self) -> str:
        '''
        Hash is the combination of the:
            1) Dataset
            2) Transformers
            3) Transformer Params
            4) Pipeline Config
        '''
        dataset_hash = self.dataset.hash_ or self.dataset._hash()
        transformers = self.get_transformers()
        transformer_params = self.get_params(params_only=True)
        pipeline_config = self.config

        return self.custom_hasher((dataset_hash, transformers, transformer_params, pipeline_config))

    def save(self, **kwargs) -> None:
        '''
        Extend parent function with a few additional save routines

        1) save params
        2) save transformer metadata
        3) features
        '''
        self.assert_dataset('Must set dataset before saving')
        self.assert_fitted('Must fit pipeline before saving')

        self.params = self.get_params(params_only=True, **kwargs)
        self.metadata_['transformers'] = self.get_transformers()
        self.metadata_['feature_names'] = self.get_feature_names()

        # Skip file-based persistence if there are no transformers
        if not self.get_transformers():
            self.has_external_files = False

        super(AbstractPipeline, self).save(**kwargs)

        # Sqlalchemy updates relationship references after save so reload class
        self.dataset.load(load_externals=False)

    def load(self, **kwargs) -> None:
        '''
        Extend main load routine to load relationship class
        '''
        super(AbstractPipeline, self).load(**kwargs)

        # Create dummy pipeline if one wasnt saved
        if not self.has_external_files:
            self._external_file = self._create_external_pipeline([], **self.params)

        # By default dont load data unless it actually gets used
        self.dataset.load(load_externals=False)

    '''
    Data Accessors
    '''

    def split_dataset(self) -> None:
        '''
        Method to create a cached reference to the projected data (cant use dataset
        directly in case of mutation concerns)

        Non-split mixin class. Returns full dataset for any split name
        '''
        default_split = IdentityProjectedDatasetSplit(dataset=self.dataset, split=None)
        # use a single reference to avoid duplicating on different key searches
        self._dataset_splits = SplitContainer(
            default_factory=lambda: default_split
        )

    def get_dataset_split(self, split: Optional[str] = None) -> ProjectedDatasetSplit:
        '''
        Get specific dataset split
        Assumes a ProjectedDatasetSplit object (`simpleml.pipelines.projected_splits.ProjectedDatasetSplit`)
        is returned. Inherit or implement similar expected attributes to replace

        Uses internal `self._dataset_splits` as the split container - assumes
        dictionary like itemgetter
        '''
        if not hasattr(self, '_dataset_splits') or self._dataset_splits is None:
            self.split_dataset()

        return self._dataset_splits[split]

    def get_split_names(self) -> List[str]:
        if not hasattr(self, '_dataset_splits') or self._dataset_splits is None:
            self.split_dataset()
        return list(self._dataset_splits.keys())

    def X(self, split: Optional[str] = None) -> Any:
        '''
        Get X for specific dataset split
        '''
        return self.get_dataset_split(split=split).X

    def y(self, split: Optional[str] = None) -> Any:
        '''
        Get labels for specific dataset split
        '''
        return self.get_dataset_split(split=split).y

    def _filter_fit_params(self, split: ProjectedDatasetSplit) -> Dict[str, Any]:
        '''
        Helper to filter unsupported fit params from dataset splits
        '''
        supported_fit_params = {}

        # Ensure input compatibility with split object
        fit_params = inspect.signature(self.external_pipeline.fit).parameters
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

        return supported_fit_params

    def fit(self):
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
        split = self.get_dataset_split(split=TRAIN_SPLIT)
        supported_fit_params = self._filter_fit_params(split)
        self.external_pipeline.fit(**supported_fit_params)
        self.fitted = True

        return self

    def transform(self, X: Any, **kwargs) -> Any:
        '''
        Main transform routine - routes to generator or regular method depending
        on the flag

        :param return_generator: boolean, whether to use the transformation method
        that returns a generator object or the regular transformed input
        :param return_sequence: boolean, whether to use method that returns a
        `keras.utils.sequence` object to play nice with keras models
        '''
        self.assert_fitted('Must fit pipeline before transforming')
        return self._transform(X, **kwargs)

    def _transform(self, X: Any, dataset_split: Optional[str] = None) -> Any:
        '''
        Pass through method to external pipeline

        :param X: dataframe/matrix to transform, if None, use internal dataset
        :rtype: Split object if no dataset passed (X is Null). Otherwise matrix
            return of input X
        '''
        if X is None:  # Retrieve dataset split
            split = self.get_dataset_split(split=dataset_split)
            if split.X is None or (isinstance(split.X, pd.DataFrame) and split.X.empty):
                output = None  # Skip transformations on empty dataset
            else:
                output = self.external_pipeline.transform(split.X)

            # Return input with X replaced by output (transformed X)
            # Contains y or other named inputs to propagate downstream
            return Split(X=output, **{k: v for k, v in split.items() if k != 'X'})

        return self.external_pipeline.transform(X)

    def fit_transform(self, **kwargs) -> Any:
        '''
        Wrapper for fit and transform methods
        ASSUMES only applies to default (train) split
        '''
        self.fit()
        return self.transform(X=None, **kwargs)

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

    def get_feature_names(self) -> List[str]:
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
