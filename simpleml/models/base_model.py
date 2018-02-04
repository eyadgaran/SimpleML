from simpleml.persistables.base_persistable import BasePersistable, GUID
from sqlalchemy import Column, String, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from abc import abstractmethod

__author__ = 'Elisha Yadgaran'


class BaseModel(BasePersistable):
    '''
    Base class for all Model objects. Defines the required
    parameters for versioning and all other metadata can be
    stored in the arbitrary metadata field

    -------
    Schema
    -------
    section: organizational attribute to manage many models pertaining to a single grouping
        ex: partitioning on an attribute and training an individual model for
        each instance (instead of one model with the attribute as a feature)
    version: string version 'x.y.z' of the model
    version_description: description that explains what is new or different about this version
    pipeline_id: foreign key relation to the pipeline used to transform input to the model
        (training is also dependent on originating dataset but scoring only needs access to the pipeline)
    params: model parameter metadata for easy insight into hyperparameters across trainings
    feature_metadata: metadata insight into resulting features and importances
    '''
    __tablename__ = 'models'

    section = Column(String, default='default')
    version = Column(String, nullable=False)
    version_description = Column(String, default='')

    # Only dependency is the pipeline (to score in production)
    pipeline_id = Column(GUID, ForeignKey("pipelines.id"))
    pipeline = relationship("BasePipeline")

    # Additional model specific metadata
    params = Column(JSONB, default={})
    feature_metadata = Column(JSONB, default={})

    def __init__(self, version, version_description=None,
                 section=None, *args, **kwargs):
        super(BaseModel, self).__init__(*args, **kwargs)
        self.section = section
        self.version = version
        self.version_description = version_description

        # Instantiate model
        self._external_model = self._create_external_model(*args, **kwargs)

    @property
    def external_model(self):
        '''
        All model objects are going to require some filebase persisted object

        Wrapper around whatever underlying class is desired
        (eg sklearn or keras)
        '''
        return self._external_model

    @abstractmethod
    def _create_external_model(self, *args, **kwargs):
        '''
        Abstract method for each subclass to implement

        should return the desired model object
        '''

    def add_pipeline(self, pipeline):
        '''
        Setter method for pipeline used
        '''
        self.pipeline = pipeline

    def save(self, *args, **kwargs):
        '''
        Extend parent function with a few additional save routines

        1) save params
        2) save feature metadata
        '''
        self.params = self.get_params(*args, **kwargs)
        self.feature_metadata = self.get_feature_metadata(*args, **kwargs)

        super(BaseModel, self).save()

    @abstractmethod
    def _save_external_files(self):
        '''
        Each subclass needs to instantiate a save routine to persist
        any other required files
        '''

    @abstractmethod
    def _load_external_files(self):
        '''
        Each subclass needs to instantiate a load routine to read in
        any other required files
        '''

    def fit(self, X, y=None, *args, **kwargs):
        '''
        Pass through method to external model
        '''
        return self.external_model.fit(X, y, *args, **kwargs)

    def predict(self, X):
        '''
        Pass through method to external model
        '''
        return self.external_model.predict(X)

    def fit_predict(self, X, y=None, *args, **kwargs):
        '''
        Pass through method to external model
        '''
        return self.external_model.fit_predict(X, y, *args, **kwargs)

    def get_params(self, *args, **kwargs):
        '''
        Pass through method to external model
        '''
        return self.external_model.get_params(*args, **kwargs)

    def set_params(self, **params):
        '''
        Pass through method to external model
        '''
        return self.external_model.set_params(**params)

    def score(self, X, y=None, *args, **kwargs):
        '''
        Pass through method to external model
        '''
        return self.external_model.score(X, y, *args, **kwargs)

    @abstractmethod
    def get_feature_metadata(self):
        '''
        Abstract method for each model to define

        Should return a dict of feature information (importance, coefficients...)
        '''
