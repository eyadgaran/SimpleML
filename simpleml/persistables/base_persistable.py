from sqlalchemy import MetaData, Column, func, String, Boolean, Integer,\
    UniqueConstraint, Index
from sqlalchemy.dialects.postgresql import JSONB
from simpleml.persistables.meta_registry import MetaRegistry, SIMPLEML_REGISTRY
from simpleml.persistables.guid import GUID
from simpleml.persistables.base_sqlalchemy import BaseSQLAlchemy
import uuid
from abc import abstractmethod

__author__ = 'Elisha Yadgaran'


class BasePersistable(BaseSQLAlchemy):
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
    registered_name: class name of object defined when importing
        Can be used for the drag and drop GUI - also for prescribing training config
    author: creator
    name: friendly name - primary way of tracking evolution of "same" object over time
    version: autoincrementing id of "friendly name"

    # Persistence of fitted states
    has_external_files = boolean field to signify presence of saved files not in db
    external_filename = path to file, relative to base simpleml folder (default ~/.simpleml)

    metadata: Generic JSON store for random attributes
    '''
    __abstract__ = True
    __metaclass__ = MetaRegistry
    # Uses main (public) schema
    metadata = MetaData()

    # Use random uuid for graceful distributed instantiation
    # also allows saved objects to include id in filename (before db persistence)
    id = Column(GUID, primary_key=True, default=uuid.uuid4)

    # Specific metadata for versioning and comparison
    # Use hash for code/data content for referencing similar objects
    # Use registered name for internal object pointer - internal code can
    # still get updated between trainings (hence hash)
    # TODO: figure out how to hash objects in a way that signifies code content
    hash_ = Column('hash', String, nullable=False)
    registered_name = Column(String, nullable=False)
    author = Column(String, default='default')
    name = Column(String, nullable=False)
    version = Column(Integer, nullable=False)

    # Persistence of fitted states
    has_external_files = Column(Boolean, default=False)
    external_filename = Column(String, nullable=True)

    # Generic store and metadata for all child objects
    metadata_ = Column('metadata', JSONB, default={})

    __table_args__ = (
        # Unique constraint for versioning
        UniqueConstraint('name', 'version', name='name_version_unique'),
        # Index for searching through friendly names
        Index('name_index', 'name'),
     )

    def __init__(self, name, has_external_files=False,
                 author=None, metadata_={}, *args, **kwargs):
        # Initialize values expected to exist at time of instantiation
        self.registered_name = self.__class__.__name__
        self.id = uuid.uuid4()
        self.author = author
        self.name = name
        self.has_external_files = has_external_files
        self.metadata_ = metadata_

    def save(self):
        '''
        Each subclass needs to instantiate a save routine to persist to the
        database and any other required filestore

        sqlalchemy_mixins supports active record style TableModel.save()
        so can still call super(BasePersistable, self).save()
        '''
        if self.has_external_files:
            self._save_external_files()

        # Hash contents upon save
        self.hash_ = self._hash()

        # Get the latest version for this "friendly name"
        self.version = self._get_latest_version()

        super(BasePersistable, self).save()

    def _save_external_files(self):
        '''
        Each subclass needs to instantiate a save routine to persist
        any other required files

        Opt not to use abstractmethod for default behavior of no external files
        '''
        raise NotImplementedError

    @abstractmethod
    def _hash(self):
        '''
        Each subclass should implement a hashing routine to uniquely AND consistently
        identify the object contents. Consistency is important to ensure ability
        to assert identity across code definitions
        '''

    def _get_latest_version(self):
        '''
        Versions should be autoincrementing for each object (constrained over
        friendly name). Executes a database lookup and increments..
        '''
        last_version = self.__class__.query_by(
            func.max(self.__class__.version)
        ).filter(
            self.__class__.name == self.name
        ).scalar()

        if last_version is None:
            last_version = 0

        return last_version + 1

    def load(self):
        '''
        Counter operation for save
        Needs to load any file and db objects

        Class definition is stored by registered_name param and
        Pickled objects are stored in external_filename param
        '''

        # Lookup appropriate class and reinstantiate
        self.__class__ = self._load_class()

        if self.has_external_files():
            self._load_external_files()

    def _load_class(self):
        '''
        Wrapper function to call global registry of all imported class names
        '''
        return SIMPLEML_REGISTRY.get(self.registered_name)

    def _load_external_files(self):
        '''
        Each subclass needs to instantiate a load routine to read in
        any other required files

        Opt not to use abstractmethod for default behavior of no external files
        '''
        raise NotImplementedError
