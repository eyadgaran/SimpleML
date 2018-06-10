from simpleml.persistables.base_persistable import BasePersistable
from simpleml.pipelines.external_pipelines import DefaultPipeline, SklearnPipeline
from simpleml.persistables.binary_blob import BinaryBlob
from simpleml.utils.errors import PipelineError
from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import JSONB
import dill as pickle

__author__ = 'Elisha Yadgaran'


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
                 *args, **kwargs):
        super(BasePipeline, self).__init__(
            has_external_files=has_external_files, *args, **kwargs)

        # Instantiate pipeline
        self._external_pipeline = self._create_external_pipeline(transformers, *args, **kwargs)
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
                                  *args, **kwargs):
        '''
        should return the desired pipeline object

        :param pipeline_class: str of class to use, can be 'default' or 'sklearn'
        '''
        if pipeline_class == 'default':
            return DefaultPipeline(transformers)
        elif pipeline_class == 'sklearn':
            return SklearnPipeline(transformers)
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

    def save(self, *args, **kwargs):
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

        self.params = self.get_params(*args, **kwargs)
        self.metadata_['transformers'] = self.get_transformers()
        self.metadata_['feature_names'] = self.get_feature_names()

        super(BasePipeline, self).save(*args, **kwargs)

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
        Shared method to save dataframe into a new table with name = GUID

        Hardcoded to only store in database so overwrite to use pickled
        objects or other storage mechanism
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

    def fit(self, *args, **kwargs):
        '''
        Pass through method to external pipeline
        '''
        if self.dataset is None:
            raise PipelineError('Must set dataset before fitting')

        output = self.external_pipeline.fit(
            self.dataset.X, self.dataset.y, *args, **kwargs)
        self._fitted = True
        return output

    def transform(self, *args, **kwargs):
        '''
        Pass through method to external pipeline
        '''
        if self.dataset is None:
            raise PipelineError('Must set dataset before transforming')

        if not self._fitted:
            raise PipelineError('Must fit pipeline before transforming')

        return self.external_pipeline.transform(
            self.dataset.X, self.dataset.y, *args, **kwargs)

    def fit_transform(self, *args, **kwargs):
        '''
        Pass through method to external pipeline
        '''
        if self.dataset is None:
            raise PipelineError('Must set dataset before fitting')

        output = self.external_pipeline.fit_transform(
            self.dataset.X, self.dataset.y, *args, **kwargs)
        self._fitted = True

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
