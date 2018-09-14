'''
Base class for sqlalchemy
'''

__author__ = 'Elisha Yadgaran'


from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, DateTime, func
from sqlalchemy_mixins import AllFeaturesMixin


Base = declarative_base()


class BaseSQLAlchemy(Base, AllFeaturesMixin):
    '''
    Base class for all SimpleML database objects. Defaults to PostgreSQL
    but can be swapped out for any supported SQLAlchemy backend.

    Takes advantage of sqlalchemy-mixins to enable active record operations
    (TableModel.save(), create(), where(), destroy())

    Added some inheritable convenience methods

    -------
    Schema
    -------
    created_timestamp: Server time on insert
    modified_timestamp: Server time on update
    '''
    __abstract__ = True

    created_timestamp = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    modified_timestamp = Column(DateTime(timezone=True), server_onupdate=func.now())

    @classmethod
    def filter(cls, *filters):
        return cls._session.query(cls).filter(*filters)

    @classmethod
    def query_by(cls, *queries):
        return cls._session.query(*queries)
