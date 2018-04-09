'''
Optional module to persist pickled objects in database instead of filesystem
'''

from simpleml.persistables.guid import GUID
from simpleml.persistables.base_sqlalchemy import BaseSQLAlchemy
from sqlalchemy import MetaData, Column, String
import uuid

__author__ = 'Elisha Yadgaran'

BINARY_STORAGE_SCHEMA = 'BINARY'


class BinaryBlob(BaseSQLAlchemy):
    ___tablename__ = 'binary_blobs'
    # Store binary data in its own schema
    metadata = MetaData(schema=BINARY_STORAGE_SCHEMA)

    id = Column(GUID, primary_key=True, default=uuid.uuid4)

    object_type = Column(String, nullable=False)
    object_id = Column(GUID, nullable=False)
    # TODO: Figure this out and think it through...
    binary_blob = Column()
