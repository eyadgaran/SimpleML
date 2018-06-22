from simpleml.persistables.guid import GUID
from simpleml.datasets.base_dataset import BaseDataset
from simpleml.persistables.dataset_storage import DatasetStorage, DATASET_SCHEMA
from simpleml.utils.errors import DatasetError
from sqlalchemy import Column, ForeignKey, UniqueConstraint, Index
from sqlalchemy.orm import relationship
import pandas as pd


__author__ = 'Elisha Yadgaran'


class BaseProcessedDataset(BaseDataset):
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

    def build_dataframe(self):
        '''
        Transform raw dataset via dataset pipeline for production ready dataset
        '''
        if self.pipeline is None:
            raise DatasetError('Must set pipeline before building dataframe')

        X, y, administrative = self.pipeline.transform(X=None, return_y=True, return_administrative=True)

        if y is None:
            y = pd.DataFrame()

        if administrative is None:
            administrative = pd.DataFrame()

        self.metadata_['label_columns'] = y.columns.tolist()
        self.metadata_['administrative_columns'] = administrative.columns.tolist()
        self._dataframe = pd.concat([X, y, administrative], axis=1)

    def save(self, **kwargs):
        '''
        Extend parent function with a few additional save routines
        '''
        if self.pipeline is None:
            raise DatasetError('Must set dataset pipeline before saving')

        super(BaseProcessedDataset, self).save(**kwargs)

        # Sqlalchemy updates relationship references after save so reload class
        self.pipeline.load(load_externals=False)

    def load(self, **kwargs):
        '''
        Extend main load routine to load relationship class
        '''
        super(BaseProcessedDataset, self).load(**kwargs)

        # By default dont load data unless it actually gets used
        self.pipeline.load(load_externals=False)
