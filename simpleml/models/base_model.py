from simpleml.persistables.base_persistable import BasePersistable, GUID
from simpleml.utils.errors import ModelError
from simpleml.persistables.binary_blob import BinaryBlob
from simpleml.datasets.base_dataset import TRAIN_CATEGORY
from sqlalchemy import Column, ForeignKey, UniqueConstraint, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
import dill as pickle
import logging


__author__ = 'Elisha Yadgaran'


LOGGER = logging.getLogger(__name__)


class BaseModel(BasePersistable):
    '''
    Base class for all Model objects. Defines the required
    parameters for versioning and all other metadata can be
    stored in the arbitrary metadata field

    -------
    Schema
    -------
    pipeline_id: foreign key relation to the pipeline used to transform input to the model
        (training is also dependent on originating dataset but scoring only needs access to the pipeline)
    params: model parameter metadata for easy insight into hyperparameters across trainings
    feature_metadata: metadata insight into resulting features and importances
    '''
    __tablename__ = 'models'

    # Only dependency is the pipeline (to score in production)
    pipeline_id = Column(GUID, ForeignKey("pipelines.id"))
    pipeline = relationship("BaseProductionPipeline", enable_typechecks=False)

    # Additional model specific metadata
    params = Column(JSONB, default={})
    feature_metadata = Column(JSONB, default={})

    __table_args__ = (
        # Unique constraint for versioning
        UniqueConstraint('name', 'version', name='model_name_version_unique'),
        # Index for searching through friendly names
        Index('model_name_index', 'name'),
     )


    def __init__(self, has_external_files=True, **kwargs):
        super(BaseModel, self).__init__(
            has_external_files=has_external_files, **kwargs)

        # Instantiate model
        self._external_model = self._create_external_model(**kwargs)
        # Initialize as unfitted
        self._fitted = False

    @property
    def external_model(self):
        '''
        All model objects are going to require some filebase persisted object

        Wrapper around whatever underlying class is desired
        (eg sklearn or keras)
        '''
        if self.unloaded_externals:
            self._load_external_files()

        return self._external_model

    def _create_external_model(self, **kwargs):
        '''
        Abstract method for each subclass to implement

        should return the desired model object
        '''
        raise NotImplementedError

    def add_pipeline(self, pipeline):
        '''
        Setter method for pipeline used
        '''
        self.pipeline = pipeline

    def _hash(self):
        '''
        Hash is the combination of the:
            1) Pipeline
            2) Model
            3) Params
        '''
        pipeline_hash = self.pipeline.hash_ or self.pipeline._hash()
        model = self.external_model
        params = self.get_params()

        return hash(self.custom_hasher((pipeline_hash, model, params)))

    def save(self, **kwargs):
        '''
        Extend parent function with a few additional save routines

        1) save params
        2) save feature metadata
        '''
        if self.pipeline is None:
            raise ModelError('Must set pipeline before saving')

        if not self._fitted:
            raise ModelError('Must fit model before saving')

        self.params = self.get_params(**kwargs)
        self.feature_metadata = self.get_feature_metadata(**kwargs)

        super(BaseModel, self).save(**kwargs)

        # Sqlalchemy updates relationship references after save so reload class
        self.pipeline.load(load_externals=False)

    def load(self, **kwargs):
        '''
        Extend main load routine to load relationship class
        '''
        super(BaseModel, self).load(**kwargs)

        # By default dont load data unless it actually gets used
        self.pipeline.load(load_externals=False)

    def _save_external_files(self):
        '''
        Shared method to save model into binary schema

        Hardcoded to only store pickled objects in database so overwrite to use
        other storage mechanism
        '''
        pickled_file = pickle.dumps(self.external_model)
        pickled_record = BinaryBlob.create(
            object_type='MODEL', object_id=self.id, binary_blob=pickled_file)
        self.filepaths = {"pickled": [str(pickled_record.id)]}

    def _load_external_files(self):
        '''
        Shared method to load model from database

        Hardcoded to only pull from pickled so overwrite to use
        other storage mechanism
        '''
        pickled_id = self.filepaths['pickled'][0]
        pickled_file = BinaryBlob.find(pickled_id).binary_blob
        self._external_model = pickle.loads(pickled_file)

        # can only be saved if fitted, so restore state
        self._fitted = True

        # Indicate externals were loaded
        self.unloaded_externals = False

    def fit(self, **kwargs):
        '''
        Pass through method to external model after running through pipeline
        '''
        if self.pipeline is None:
            raise ModelError('Must set pipeline before fitting')

        if self._fitted:
            LOGGER.warning('Cannot refit model, skipping operation')
            return self

        # Explicitly fit only on train split
        X, y = self.pipeline.transform(X=None, sample_category=TRAIN_CATEGORY, return_y=True)
        self.external_model.fit(X, y.squeeze(), **kwargs)
        self._fitted = True

        return self

    def predict(self, X, **kwargs):
        '''
        Pass through method to external model after running through pipeline
        '''
        if not self._fitted:
            raise ModelError('Must fit model before predicting')

        transformed = self.pipeline.transform(X, **kwargs)

        return self.external_model.predict(transformed)

    def fit_predict(self, **kwargs):
        '''
        Wrapper for fit and predict methods
        '''
        self.fit(**kwargs)
        # Pass X as none to cascade using internal dataset for X
        # Assumes only applies to training split
        return self.predict(X=None, sample_category=TRAIN_CATEGORY, **kwargs)

    def get_params(self, **kwargs):
        '''
        Pass through method to external model
        '''
        return self.external_model.get_params(**kwargs)

    def set_params(self, **params):
        '''
        Pass through method to external model
        '''
        return self.external_model.set_params(**params)

    def score(self, X, y=None, **kwargs):
        '''
        Pass through method to external model
        '''
        return self.external_model.score(X, y, **kwargs)

    def get_feature_metadata(self, **kwargs):
        '''
        Abstract method for each model to define

        Should return a dict of feature information (importance, coefficients...)
        '''
        return self.external_model.get_feature_metadata(**kwargs)
