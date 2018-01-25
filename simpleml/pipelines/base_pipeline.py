from simpleml.persistables.base_persistable import BasePersistable
from sqlalchemy import Column, String, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

__author__ = 'Elisha Yadgaran'


class BasePipeline(BasePersistable):
    '''
    Base class for all Pipelines objects.

    -------
    Schema
    -------
    transformers: json with list of transformer objects
    dataset_id: foreign key relation to the dataset used as input
    '''
    __tablename__ = 'pipelines'

    transformers = Column(JSONB, default={})

    dataset_id = Column(String, ForeignKey("datasets.id"))
    dataset = relationship("BaseDataset")
