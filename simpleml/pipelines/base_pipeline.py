from simpleml.persistables.base_persistable import BasePersistable, GUID
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
    __abstract__ = True

    transformers = Column(JSONB, default={})
