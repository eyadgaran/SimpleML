from simpleml.persistables.base_persistable import Persistable, GUID, MutableJSON
from simpleml.registries import MetricRegistry
from simpleml.utils.errors import MetricError
from sqlalchemy import Column, ForeignKey, UniqueConstraint, Index, func
from sqlalchemy.orm import relationship
from future.utils import with_metaclass
import logging

__author__ = 'Elisha Yadgaran'


LOGGER = logging.getLogger(__name__)


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

    values = Column(MutableJSON, nullable=False)

    object_type = 'METRIC'

    def add_dataset(self, dataset):
        '''
        Setter method for dataset used
        '''
        self.dataset = dataset

    def add_model(self, model):
        '''
        Setter method for model used
        '''
        self.model = model

    def _hash(self):
        '''
        Hash is the combination of the:
            1) Model
            2) Dataset (optional)
            3) Metric
            4) Config
        '''
        model_hash = self.model.hash_ or self.model._hash()
        if self.dataset is not None:
            dataset_hash = self.dataset.hash_ or self.dataset._hash()
        else:
            dataset_hash = None
        metric = self.__class__.__name__
        config = self.config

        return self.custom_hasher((model_hash, dataset_hash, metric, config))

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

    def _get_pipeline_split(self, column: str, split: str, **kwargs):
        '''
        For special case where dataset is the same as the model's dataset, the
        dataset splits can refer to the pipeline imposed splits, not the inherent
        dataset's splits. Use the pipeline split then
        ex: RandomSplitPipeline on NoSplitDataset evaluating "in_sample" performance
        '''
        return getattr(self.model.pipeline.get_dataset_split(split=split, **kwargs), column)

    def _get_dataset_split(self, **kwargs):
        '''
        Default accessor for dataset data. REFERS TO RAW DATASETS
        not the pipelines superimposed. That means that datasets that do not
        define explicit splits will have no notion of downstream splits
        (e.g. RandomSplitPipeline)
        '''
        return self.dataset.get(**kwargs)

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

    # Dependencies are model and dataset
    model_id = Column(GUID, ForeignKey("models.id", name="metrics_model_id_fkey"))
    model = relationship('Model', enable_typechecks=False)
    dataset_id = Column(GUID, ForeignKey("datasets.id", name="metrics_dataset_id_fkey"))
    dataset = relationship('Dataset', enable_typechecks=False)

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
