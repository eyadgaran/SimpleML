'''
Optional module to persist pickled objects in database instead of filesystem
'''

from simpleml.persistables.base_persistable import Base, AllFeaturesMixin, GUID
from sqlalchemy import Column, DateTime, func, String
import uuid

__author__ = 'Elisha Yadgaran'


class BinaryBlob(Base, AllFeaturesMixin):
    ___tablename__ = 'binary_blobs'

    id = Column(GUID, primary_key=True, default=uuid.uuid4)

    object_type = Column(String, nullable=False)
    object_id = Column(GUID, nullable=False)
    # TODO: Figure this out and think it through...
    binary_blob = Column()

    created_timestamp = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    modified_timestamp = Column(DateTime(timezone=True), server_onupdate=func.now())
