'''
Optional module to persist pickled objects in database instead of filesystem
'''

from simpleml.persistables.sqlalchemy_types import GUID
from simpleml.persistables.base_sqlalchemy import BaseSQLAlchemy
from sqlalchemy import MetaData, Column, String, LargeBinary, event, DDL
import uuid

__author__ = 'Elisha Yadgaran'

BINARY_STORAGE_SCHEMA = 'BINARY'


class BinaryBlob(BaseSQLAlchemy):
    __tablename__ = 'binary_blobs'
    # Store binary data in its own schema
    metadata = MetaData(schema=BINARY_STORAGE_SCHEMA)

    id = Column(GUID, primary_key=True, default=uuid.uuid4)

    object_type = Column(String, nullable=False)
    object_id = Column(GUID, nullable=False)
    # TODO: Figure this out and think it through...
    binary_blob = Column(LargeBinary)

    event.listen(metadata, 'before_create', DDL('''CREATE SCHEMA IF NOT EXISTS "{}";'''.format(BINARY_STORAGE_SCHEMA)))
