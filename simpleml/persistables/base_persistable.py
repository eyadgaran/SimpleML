from sqlalchemy.ext.declarative import DeclaritiveBase
from sqlalchemy import MetaData, Column, func, DateTime
from sqlalchemy_mixins import AllFeaturesMixin
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import TypeDecorator, CHAR
from sqlalchemy.dialects.postgresql import UUID
import uuid

__author__ = 'Elisha Yadgaran'


metadata = MetaData()
Base = DeclaritiveBase(metadata=metadata)


class GUID(TypeDecorator):
    """Platform-independent GUID type.

    Uses PostgreSQL's UUID type, otherwise uses
    CHAR(32), storing as stringified hex values.

    http://docs.sqlalchemy.org/en/latest/core/custom_types.html
    """
    impl = CHAR

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(UUID())
        else:
            return dialect.type_descriptor(CHAR(32))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return str(value)
        else:
            if not isinstance(value, uuid.UUID):
                return "%.32x" % uuid.UUID(value).int
            else:
                # hexstring
                return "%.32x" % value.int

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        else:
            if not isinstance(value, uuid.UUID):
                value = uuid.UUID(value)
            return value


class BasePersistable(Base, AllFeaturesMixin):
    '''
    Base class for all SimpleML database objects. Defaults to PostgreSQL
    but can be swapped out for any supported SQLAlchemy backend.

    Takes advantage of sqlalchemy-mixins to enable active record operations
    (TableModel.save(), create(), where(), destroy())
    '''
    __abstract__ = True

    # Use random uuid for graceful distributed instantiation
    id = Column(GUID, primary_key=True, default=uuid.uuid4())

    # Generic store and metadata for all child objects
    metadata = Column(JSONB, default={})
    created_timestamp = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    modified_timestamp = Column(DateTime(timezone=True), server_onupdate=func.now())
