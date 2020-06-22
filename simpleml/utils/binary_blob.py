'''
Optional module to persist pickled objects in database instead of filesystem
'''

from simpleml.persistables.sqlalchemy_types import GUID
from simpleml.persistables.base_sqlalchemy import BinaryStorageSqlalchemy
from sqlalchemy import Column, String, LargeBinary
import uuid
import logging

__author__ = 'Elisha Yadgaran'

LOGGER = logging.getLogger(__name__)


class BinaryBlob(BinaryStorageSqlalchemy):
    __tablename__ = 'binary_blobs'

    id = Column(GUID, primary_key=True, default=uuid.uuid4)

    object_type = Column(String, nullable=False)
    object_id = Column(GUID, nullable=False)
    # TODO: Figure this out and think it through...
    binary_blob = Column(LargeBinary)
