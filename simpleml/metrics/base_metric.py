from simpleml.persistables.base_persistable import Persistable, GUID, JSON
from simpleml.persistables.meta_registry import MetricRegistry
from simpleml.utils.errors import MetricError
from sqlalchemy import Column, ForeignKey, UniqueConstraint, Index, func
from sqlalchemy.orm import relationship
from future.utils import with_metaclass

__author__ = 'Elisha Yadgaran'


class AbstractMetric(with_metaclass(MetricRegistry, Persistable)):
    '''
    Abstract Base class for all Metric objects

    -------
    Schema
    -------
    name: the metric name
    values: JSON object with key: value pairs for performance on test dataset
        (ex: FPR: TPR to create ROC Curve)
        Singular value metrics take the form - {'agg': value}
    '''
    __abstract__ = True

    values = Column(JSON, nullable=False)

    object_type = 'METRIC'

    def add_model(self, model):
        '''
        Setter method for model used
        '''
        self.model = model

    def _hash(self):
        '''
        Hash is the combination of the:
            1) Model
            2) Metric
            3) Config
        '''
        model_hash = self.model.hash_ or self.model._hash()
        metric = self.__class__.__name__
        config = self.config

        return self.custom_hasher((model_hash, metric, config))

    def _get_latest_version(self):
        '''
        Versions should be autoincrementing for each object (constrained over
        friendly name and model). Executes a database lookup and increments..
        '''
        last_version = self.__class__.query_by(
            func.max(self.__class__.version)
        ).filter(
            self.__class__.name == self.name,
            self.__class__.model_id == self.model.id
        ).scalar()

        if last_version is None:
            last_version = 0

        return last_version + 1

    def save(self, **kwargs):
        '''
        Extend parent function with a few additional save routines
        '''
        if self.model is None:
            raise MetricError('Must set model before saving')

        if self.values is None:
            raise MetricError('Must score metric before saving')

        super(AbstractMetric, self).save(**kwargs)

        # Sqlalchemy updates relationship references after save so reload class
        self.model.load(load_externals=False)

    def load(self, **kwargs):
        '''
        Extend main load routine to load relationship class
        '''
        super(AbstractMetric, self).load(**kwargs)

        # By default dont load data unless it actually gets used
        self.model.load(load_externals=False)

    def score(self, **kwargs):
        '''
        Abstract method for each metric to define

        Should set self.values
        '''
        raise NotImplementedError


class Metric(AbstractMetric):
    '''
    Base class for all Metric objects

    -------
    Schema
    -------
    model_id: foreign key to the model that was used to generate predictions

    TODO: Should join criteria be composite of model and dataset for multiple
        duplicate metric objects computed over different test datasets?
    '''
    __tablename__ = 'metrics'

    # Only dependency is the model (to score in production)
    model_id = Column(GUID, ForeignKey("models.id"))
    model = relationship('Model', enable_typechecks=False)

    __table_args__ = (
        # Metrics don't have the notion of versions, values should be deterministic
        # by class, model, and dataset - name should be the combination of class and dataset
        # Still exists to stay consistent with the persistables style of unrestricted duplication
        # (otherwise would be impossible to distinguish a duplicated metric -- name and model_id would be the same)

        # Unique constraint for versioning
        UniqueConstraint('name', 'model_id', 'version', name='metric_name_model_version_unique'),
        # Index for searching through friendly names
        Index('metric_name_index', 'name'),
     )
