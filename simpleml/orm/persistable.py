'''
Base class for all database tracked records, called "Persistables"
'''

__author__ = 'Elisha Yadgaran'


import logging
import uuid
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional, Type, Union

from simpleml.orm.metadata import SimplemlCoreSqlalchemy
from simpleml.orm.sqlalchemy_types import GUID, MutableJSON
from simpleml.persistables.base_persistable import Persistable
from simpleml.registries import SIMPLEML_REGISTRY
from simpleml.utils.errors import SimpleMLError
from sqlalchemy import Boolean, Column, Integer, String, func, inspect
from sqlalchemy.orm.relationships import RelationshipProperty

LOGGER = logging.getLogger(__name__)


class ORMPersistable(SimplemlCoreSqlalchemy):
    '''
    Base class for all SimpleML database objects.
    dialect can be swapped out for any supported SQLAlchemy backend.

    Takes advantage of sqlalchemy-mixins to enable active record operations
    (TableModel.save(), create(), where(), destroy())

    Uses private class attributes for internal artifact registry
    Does not need to be persisted because it gets populated on import
    (and can therefore be changed between versions)
    cls._ARTIFACT_{artifact_name} = {'save': save_attribute, 'restore': restore_attribute}

    -------
    Schema
    -------
    id: Random UUID(4). Used over auto incrementing id to minimize collision probability
        with distributed trainings and authors (especially if using central server
        to combine results across different instantiations of SimpleML)

    hash_id: Use hash of object to uniquely identify the contents at train time
    registered_name: class name of object defined when importing
        Can be used for the drag and drop GUI - also for prescribing training config
    author: creator
    project: Project objects are associated with. Useful if multiple persistables
        relate to the same project and want to be grouped (but have different names)
        also good for implementing row based security across teams
    name: friendly name - primary way of tracking evolution of "same" object over time
    version: autoincrementing id of "friendly name"
    version_description: description that explains what is new or different about this version

    # Persistence of fitted states
    has_external_files = boolean field to signify presence of saved files not in (main) db
    filepaths = JSON object with external file details
        The nested notation is because any persistable can implement multiple save
        options (with arbitrary priority) and arbitrary inputs. Simple serialization
        could have only a single string location whereas complex artifacts might have
        a list or map of filepaths

        Structure:
        {
            artifact_name: {
                'save_pattern': filepath_data
            },
            "example": {
                "disk_pickled": path to file, relative to base simpleml folder (default ~/.simpleml),
                "database": {"schema": schema, "table": table_name}, # (for files extractable with `select * from`)
                ...
            }
        }


    metadata: Generic JSON store for random attributes
    '''

    __abstract__ = True

    # Use random uuid for graceful distributed instantiation
    # also allows saved objects to include id in filename (before db persistence)
    id = Column(GUID, primary_key=True, nullable=False)

    # Specific metadata for versioning and comparison
    # Use hash for code/data content for referencing similar objects
    # Use registered name for internal object pointer - internal code can
    # still get updated between trainings (hence hash)
    # TODO: figure out how to hash objects in a way that signifies code content
    hash_ = Column('hash', String, nullable=False)
    registered_name = Column(String, nullable=False)
    author = Column(String, nullable=False)
    project = Column(String, nullable=False)
    name = Column(String, nullable=False)
    version = Column(Integer, nullable=False)
    version_description = Column(String)

    # Persistence of fitted states
    has_external_files = Column(Boolean, default=False)
    filepaths = Column(MutableJSON)

    # Generic store and metadata for all child objects
    metadata_ = Column('metadata', MutableJSON)

    @classmethod
    def save_record(cls, id: str, **kwargs) -> None:
        '''
        save overloads parent method that is called by helper methods for create/update
        '''
        attributes = inspect(cls).attrs.keys()

        # check if existing record
        record = cls.find(id)
        # create
        if record is None:
            LOGGER.debug(f'No existing record matching id {id}. Creating new one')
            cls.create(id=id, **{k: v for k, v in kwargs.items() if k in attributes})
        # update
        else:
            LOGGER.debug(f'Found existing record matching id {id}. Updating values')
            record.update(**{k: v for k, v in kwargs.items() if k in attributes})

    def load(self, load_externals: bool = False) -> Persistable:
        '''
        Counter operation for save
        Needs to load any file and db objects

        Class definition is stored by registered_name param and
        Pickled objects are stored in external_filename param

        :param load_externals: Boolean flag whether to load the external files
        useful for relationships that only need class definitions and not data
        '''

        # Lookup appropriate class and reinstantiate
        cls = self._load_class()

        persistable = cls.from_dict(**self.to_dict())

        if load_externals:
            persistable.load_external_files()

        return persistable

    def _load_class(self) -> Persistable:
        '''
        Wrapper function to call global registry of all imported class names
        '''
        cls = SIMPLEML_REGISTRY.get(self.registered_name)
        if cls is None:
            raise SimpleMLError(f'Could not find registered class for {self.registered_name}')
        return cls

    def to_dict(self):
        '''
        Utility method to inspect the orm model and return a dictionary of
        attributes -> values

        Uses the mapped attribute name, not the column name (e.g. hash_ vs hash).
        excludes relationships (to support lazy loading)
        '''
        attributes = inspect(self).mapper.all_orm_descriptors
        non_relationship_attrs = [attr for attr, v in attributes.items()
                                  if not hasattr(v, "prop")
                                  or not isinstance(v.prop, RelationshipProperty)]

        # handle type conversions from sqlalchemy types
        # def type_formatter(attr):
        #     if isinstance(attr, sqlalchemy_json.NestedMutableDict):
        #         return dict(attr)
        #     return attr

        return {attr: getattr(self, attr) for attr in non_relationship_attrs}

    @classmethod
    def get_latest_version(cls, name: str) -> int:
        '''
        Versions should be autoincrementing for each object (constrained over
        friendly name). Executes a database lookup and increments..
        '''
        last_version = cls.query_by(
            func.max(cls.version)
        ).filter(
            cls.name == name
        ).scalar()

        if last_version is None:
            last_version = 0

        return last_version + 1

    @staticmethod
    def load_reference(reference_cls: 'ORMPersistable', id: str) -> Persistable:
        record = reference_cls.find(id)
        return record.load()
