from simpleml.persistables.meta_registry import DatasetRegistry
from simpleml.persistables.guid import GUID
from simpleml.datasets.base_dataset import BaseDataset
from simpleml.datasets.pandas_mixin import PandasDatasetMixin
from simpleml.datasets.numpy_mixin import NumpyDatasetMixin
from simpleml.persistables.dataset_storage import DatasetStorage, DATASET_SCHEMA
from simpleml.utils.errors import DatasetError
from sqlalchemy import Column, ForeignKey, UniqueConstraint, Index
from sqlalchemy.orm import relationship
import pandas as pd
from future.utils import with_metaclass


__author__ = 'Elisha Yadgaran'


class AbstractBaseProcessedDataset(with_metaclass(DatasetRegistry, BaseDataset)):
    '''
    Abstract Base class for all Processed Dataset objects.
    '''
    __abstract__ = True

    @property
    def _schema(self):
        return DATASET_SCHEMA

    @property
    def _engine(self):
        return DatasetStorage.metadata.bind

    def add_pipeline(self, pipeline):
        '''
        Setter method for dataset pipeline used
        '''
        self.pipeline = pipeline

    def save(self, **kwargs):
        '''
        Extend parent function with a few additional save routines
        '''
        if self.pipeline is None:
            raise DatasetError('Must set dataset pipeline before saving')

        super(AbstractBaseProcessedDataset, self).save(**kwargs)

        # Sqlalchemy updates relationship references after save so reload class
        self.pipeline.load(load_externals=False)

    def load(self, **kwargs):
        '''
        Extend main load routine to load relationship class
        '''
        super(AbstractBaseProcessedDataset, self).load(**kwargs)

        # By default dont load data unless it actually gets used
        self.pipeline.load(load_externals=False)


class BaseProcessedDataset(AbstractBaseProcessedDataset):
    '''
    Base class for all Processed Dataset objects.

    -------
    Schema
    -------
    pipeline_id: foreign key relation to the dataset pipeline used as input
    '''
    __tablename__ = 'datasets'

    pipeline_id = Column(GUID, ForeignKey("dataset_pipelines.id"))
    pipeline = relationship("BaseDatasetPipeline", enable_typechecks=False)

    __table_args__ = (
        # Unique constraint for versioning
        UniqueConstraint('name', 'version', name='dataset_name_version_unique'),
        # Index for searching through friendly names
        Index('dataset_name_index', 'name'),
     )


# Mixin implementations for convenience
class BasePandasProcessedDataset(BaseProcessedDataset, PandasDatasetMixin):
    def build_dataframe(self):
        '''
        Transform raw dataset via dataset pipeline for production ready dataset
        '''
        if self.pipeline is None:
            raise DatasetError('Must set pipeline before building dataframe')

        X, y = self.pipeline.transform(X=None, return_y=True)

        if y is None:
            y = pd.DataFrame()

        self.config['label_columns'] = y.columns.tolist()
        self._external_file = pd.concat([X, y], axis=1)

class BaseNumpyProcessedDataset(BaseProcessedDataset, NumpyDatasetMixin):
    def build_dataframe(self):
        '''
        Transform raw dataset via dataset pipeline for production ready dataset
        '''
        if self.pipeline is None:
            raise DatasetError('Must set pipeline before building dataframe')

        X, y = self.pipeline.transform(X=None, return_y=True)

        if y is not None:
            self.config['label_columns'] = ['y']

        self._external_file = {'X': X, 'y': y}
