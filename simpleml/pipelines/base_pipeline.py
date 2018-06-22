from simpleml.persistables.base_persistable import BasePersistable
from simpleml.pipelines.external_pipelines import DefaultPipeline, SklearnPipeline
from simpleml.datasets.base_dataset import TRAIN_CATEGORY
from simpleml.persistables.binary_blob import BinaryBlob
from simpleml.utils.errors import PipelineError
from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import JSONB
import dill as pickle
import logging

__author__ = 'Elisha Yadgaran'


LOGGER = logging.getLogger(__name__)


class BasePipeline(BasePersistable):
    '''
    Base class for all Pipelines objects.

    -------
    Schema
    -------
    params: pipeline parameter metadata for easy insight into hyperparameters across trainings
    dataset_id: foreign key relation to the dataset used as input
    '''
    __abstract__ = True

    # Additional pipeline specific metadata
    params = Column(JSONB, default={})

    def __init__(self, has_external_files=True, transformers=[],
                 **kwargs):
        super(BasePipeline, self).__init__(
            has_external_files=has_external_files, **kwargs)

        # Instantiate pipeline
        self._external_pipeline = self._create_external_pipeline(transformers, **kwargs)
        # Initialize as unfitted
        self._fitted = False

    @property
    def external_pipeline(self):
        '''
        All pipeline objects are going to require some filebase persisted object

        Wrapper around whatever underlying class is desired
        (eg sklearn or native)
        '''
        if self.unloaded_externals:
            self._load_external_files()

        return self._external_pipeline

    def _create_external_pipeline(self, transformers, pipeline_class='default',
                                  **kwargs):
        '''
        should return the desired pipeline object

        :param pipeline_class: str of class to use, can be 'default' or 'sklearn'
        '''
        if pipeline_class == 'default':
            return DefaultPipeline(transformers)
        elif pipeline_class == 'sklearn':
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
        self._fitted = False

    def remove_transformer(self, name):
        '''
        Delete method for transformer step
        '''
        self.external_pipeline.remove_transformer(name)
        self._fitted = False

    def _hash(self):
        '''
        Hash is the combination of the:
            1) Dataset
            2) Transformers
            3) Params
        '''
        dataset_hash = self.dataset.hash_ or self.dataset._hash()
        transformers = self.get_transformers()
        params = self.get_params()

        return hash(self.custom_hasher((dataset_hash, transformers, params)))

    def save(self, **kwargs):
        '''
        Extend parent function with a few additional save routines

        1) save params
        2) save transformer metadata
        3) features
        '''
        if self.dataset is None:
            raise PipelineError('Must set dataset before saving')

        if not self._fitted:
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

    def _save_external_files(self):
        '''
        Shared method to save pipeline into binary schema

        Hardcoded to only store pickled objects in database so overwrite to use
        other storage mechanism
        '''
        pickled_file = pickle.dumps(self.external_pipeline)
        pickled_record = BinaryBlob.create(
            object_type='PIPELINE', object_id=self.id, binary_blob=pickled_file)
        self.filepaths = {"pickled": [str(pickled_record.id)]}

    def _load_external_files(self):
        '''
        Shared method to load pipeline from database

        Hardcoded to only pull from pickled so overwrite to use
        other storage mechanism
        '''
        pickled_id = self.filepaths['pickled'][0]
        pickled_file = BinaryBlob.find(pickled_id).binary_blob
        self._external_pipeline = pickle.loads(pickled_file)

        # can only be saved if fitted, so restore state
        self._fitted = True

        # Indicate externals were loaded
        self.unloaded_externals = False

    def fit(self, **kwargs):
        '''
        Pass through method to external pipeline
        '''
        if self.dataset is None:
            raise PipelineError('Must set dataset before fitting')

        if self._fitted:
            LOGGER.warning('Cannot refit pipeline, skipping operation')
            return self

        # Only use train fold to fit
        self.external_pipeline.fit(
            self.dataset.X(sample_category=TRAIN_CATEGORY),
            self.dataset.y(sample_category=TRAIN_CATEGORY), **kwargs)
        self._fitted = True

        return self

    def transform(self, X, sample_category=None, return_y=False,
                  return_administrative=False, **kwargs):
        '''
        Pass through method to external pipeline

        :param X: dataframe/matrix to transform, if None, use internal dataset
        :param return_y: whether to return y with output - only used if X is None
            necessary for fitting a supervised model after
        :param return_administrative: whether to return administrative with output - only used if X is None
            necessary for splitting dataset after (train/validation/test)
        '''
        if not self._fitted:
            raise PipelineError('Must fit pipeline before transforming')

        if X is None:
            output = self.external_pipeline.transform(
                self.dataset.X(sample_category=sample_category), **kwargs)

            if return_y and return_administrative:
                return output, self.dataset.y(sample_category=sample_category), self.dataset.administrative_df(sample_category=sample_category)
            elif return_y:
                return output, self.dataset.y(sample_category=sample_category)
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
        output = self.transform(X=None, sample_category=TRAIN_CATEGORY, **kwargs)

        if return_y:
            return output, self.dataset.y(sample_category=TRAIN_CATEGORY)

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
        return self.external_pipeline.get_feature_names(feature_names=self.dataset.dataframe.columns.tolist())
