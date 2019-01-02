from simpleml import TRAIN_SPLIT
from simpleml.persistables.base_persistable import BasePersistable, GUID
from simpleml.persistables.meta_registry import ModelRegistry
from simpleml.persistables.saving import AllSaveMixin
from simpleml.utils.errors import ModelError

from sqlalchemy import Column, ForeignKey, UniqueConstraint, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
import logging
from future.utils import with_metaclass


__author__ = 'Elisha Yadgaran'


LOGGER = logging.getLogger(__name__)


class AbstractBaseModel(with_metaclass(ModelRegistry, BasePersistable, AllSaveMixin)):
    '''
    Abstract Base class for all Model objects. Defines the required
    parameters for versioning and all other metadata can be
    stored in the arbitrary metadata field

    -------
    Schema
    -------
    params: model parameter metadata for easy insight into hyperparameters across trainings
    feature_metadata: metadata insight into resulting features and importances
    '''
    __abstract__ = True

    # Additional model specific metadata
    params = Column(JSONB, default={})
    feature_metadata = Column(JSONB, default={})

    def __init__(self, has_external_files=True, external_model_kwargs={}, params={}, **kwargs):
        '''
        Need to explicitly separate passthrough kwargs to external models since
        most do not support arbitrary **kwargs in the constructors
        '''
        super(AbstractBaseModel, self).__init__(
            has_external_files=has_external_files, **kwargs)

        # Instantiate model
        self.object_type = 'MODEL'
        self._external_file = self._create_external_model(**external_model_kwargs)
        self.set_params(**params)

        # Initialize as unfitted
        self.state['fitted'] = False

    @property
    def external_model(self):
        '''
        All model objects are going to require some filebase persisted object

        Wrapper around whatever underlying class is desired
        (eg sklearn or keras)
        '''
        if self.unloaded_externals:
            self._load_external_files()

        return self._external_file

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
            4) Config
        '''
        pipeline_hash = self.pipeline.hash_ or self.pipeline._hash()
        model = self.external_model.__class__.__name__
        params = self.get_params()
        config = self.config

        return self.custom_hasher((pipeline_hash, model, params, config))

    def save(self, **kwargs):
        '''
        Extend parent function with a few additional save routines

        1) save params
        2) save feature metadata
        '''
        if self.pipeline is None:
            raise ModelError('Must set pipeline before saving')

        if not self.state['fitted']:
            raise ModelError('Must fit model before saving')

        self.params = self.get_params(**kwargs)
        self.feature_metadata = self.get_feature_metadata(**kwargs)

        super(AbstractBaseModel, self).save(**kwargs)

        # Sqlalchemy updates relationship references after save so reload class
        self.pipeline.load(load_externals=False)

    def load(self, **kwargs):
        '''
        Extend main load routine to load relationship class
        '''
        super(AbstractBaseModel, self).load(**kwargs)

        # By default dont load data unless it actually gets used
        self.pipeline.load(load_externals=False)

    def _fit(self, X, y):
        '''
        Separate out actual fit call for optional overwrite in subclasses
        '''
        # Reduce dimensionality of y if it is only 1 column
        self.external_model.fit(X, y.squeeze())

    def fit(self, **kwargs):
        '''
        Pass through method to external model after running through pipeline
        '''
        if self.pipeline is None:
            raise ModelError('Must set pipeline before fitting')

        if self.state['fitted']:
            LOGGER.warning('Cannot refit model, skipping operation')
            return self

        # Explicitly fit only on train split
        X, y = self.pipeline.transform(X=None, dataset_split=TRAIN_SPLIT, return_y=True)

        self._fit(X, y)

        # Mark the state so it doesnt get refit and can now be saved
        self.state['fitted'] = True

        return self

    def _predict(self, X, **kwargs):
        '''
        Separate out actual predict call for optional overwrite in subclasses
        '''
        return self.external_model.predict(X)

    def predict(self, X, **kwargs):
        '''
        Pass through method to external model after running through pipeline
        '''
        if not self.state['fitted']:
            raise ModelError('Must fit model before predicting')

        transformed = self.pipeline.transform(X, **kwargs)

        return self._predict(transformed, **kwargs)

    def fit_predict(self, **kwargs):
        '''
        Wrapper for fit and predict methods
        '''
        self.fit(**kwargs)
        # Pass X as none to cascade using internal dataset for X
        # Assumes only applies to training split
        return self.predict(X=None, dataset_split=TRAIN_SPLIT, **kwargs)

    def get_labels(self, dataset_split=None):
        '''
        Wrapper method to return labels from dataset
        '''
        return self.pipeline.get_dataset_split(dataset_split)[1]

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
        return self.external_model.get_feature_metadata(features=self.pipeline.get_feature_names, **kwargs)


class BaseModel(AbstractBaseModel):
    '''
    Base class for all Model objects. Defines the required
    parameters for versioning and all other metadata can be
    stored in the arbitrary metadata field

    -------
    Schema
    -------
    pipeline_id: foreign key relation to the pipeline used to transform input to the model
        (training is also dependent on originating dataset but scoring only needs access to the pipeline)
    '''
    __tablename__ = 'models'

    # Only dependency is the pipeline (to score in production)
    pipeline_id = Column(GUID, ForeignKey("pipelines.id"))
    pipeline = relationship("BaseProductionPipeline", enable_typechecks=False)

    __table_args__ = (
        # Unique constraint for versioning
        UniqueConstraint('name', 'version', name='model_name_version_unique'),
        # Index for searching through friendly names
        Index('model_name_index', 'name'),
     )
