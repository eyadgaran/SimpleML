"""
Base class for sqlalchemy table models. Defaults some opinionated fields for
all inherited tables.
"""

__author__ = "Elisha Yadgaran"

import logging

from sqlalchemy import Column, DateTime, event, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy_mixins import AllFeaturesMixin

Base = declarative_base()
LOGGER = logging.getLogger(__name__)


class BaseSQLAlchemy(Base, AllFeaturesMixin):
    """
    Base class for sqlalchemy table models. Defaults some opinionated fields for
    all inherited tables.

    A sqlalchemy.MetaData object needs to be defined on each table/table group
    and then initialized as part of a session to be attached to a database.

    Premixes the following base classes for all table models:
    - sqlalchemy-mixins (AllFeatureMixin)

    Takes advantage of sqlalchemy-mixins to enable active record operations
    (TableModel.save(), create(), where(), destroy())

    Added some inheritable convenience methods

    -------
    Schema
    -------
    created_timestamp: Server time on insert
    modified_timestamp: Server time on update
    """

    __abstract__ = True

    created_timestamp = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    modified_timestamp = Column(DateTime(timezone=True), server_onupdate=func.now())

    @classmethod
    def filter(cls, *filters):
        return cls._session.query(cls).filter(*filters)

    @classmethod
    def query_by(cls, *queries):
        return cls._session.query(*queries)


# Sqlalchemy registered listener to update fields on all table model changes
# (only registered in code so external db modifications will not trigger)
@event.listens_for(BaseSQLAlchemy, "before_update", propagate=True)
def _receive_before_update(mapper, connection, target):
    """Listen for updates and update `modified_timestamp` column."""
    target.modified_timestamp = func.now()
