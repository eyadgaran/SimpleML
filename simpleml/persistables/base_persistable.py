from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import MetaData, Column, func, DateTime, String, Boolean
from sqlalchemy_mixins import AllFeaturesMixin
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import TypeDecorator, CHAR
from sqlalchemy.dialects.postgresql import UUID
import uuid

__author__ = 'Elisha Yadgaran'


metadata = MetaData()
Base = declarative_base(metadata=metadata)


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

    -------
    Schema
    -------
    id: Random UUID(4). Used over auto incrementing id to minimize collision probability
        with distributed trainings and authors (especially if using central server
        to combine results across different instantiations of SimpleML)

    hash_id: Use hash of object to uniquely identify the contents at train time
    registered_name: internal name of object used when instantiating the class
        Can be used for the drag and drop GUI

    # Persistence of fitted states
    has_external_files = boolean field to signify presence of saved files not in db
    external_filename = path to file, relative to base simpleml folder (default ~/.simpleml)

    metadata: Generic JSON store for random attributes
    created_timestamp: Server time on insert
    modified_timestamp: Server time on update
    '''
    __abstract__ = True

    # Use random uuid for graceful distributed instantiation
    # also allows saved objects to include id in filename (before db persistence)
    id = Column(GUID, primary_key=True, default=uuid.uuid4())

    # Specific metadata for versioning and comparison
    # Use hash for code/data content for referencing similar objects
    # Use registered name for internal object pointer - internal code can
    # still get updated between trainings (hence hash)
    # TODO: figure out how to hash objects in a way that signifies code content
    hash_id = Column(String, nullable=False)
    registered_name = Column(String, nullable=False)

    # Persistence of fitted states
    has_external_files = Column(Boolean, default=False)
    external_filename = Column(String, nullable=True)

    # Generic store and metadata for all child objects
    metadata_ = Column('metadata', JSONB, default={})
    created_timestamp = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    modified_timestamp = Column(DateTime(timezone=True), server_onupdate=func.now())
