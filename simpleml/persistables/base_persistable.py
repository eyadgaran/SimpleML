from sqlalchemy import MetaData, Column, func, String, Boolean, Integer, BigInteger
from sqlalchemy.dialects.postgresql import JSONB
from simpleml.persistables.meta_registry import MetaRegistry, SIMPLEML_REGISTRY
from simpleml.persistables.guid import GUID
from simpleml.persistables.base_sqlalchemy import BaseSQLAlchemy
from simpleml.utils.library_versions import INSTALLED_LIBRARIES
import uuid
from abc import abstractmethod
import copy
import pandas as pd
from pandas.util import hash_pandas_object


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
    version_description: description that explains what is new or different about this version

    # Persistence of fitted states
    has_external_files = boolean field to signify presence of saved files not in (main) db
    filepaths = JSON object with external file details
        Structure:
        {
            "disk": [
                path to file, relative to base simpleml folder (default ~/.simpleml),
                ...
            ],
            "database": [
                (schema, table_name), (for files extractable with `select * from`)
                ....
            ],
            "pickled": [
                guid, (for files in binary blobs)
                ...
            ]
        }

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
    hash_ = Column('hash', BigInteger, nullable=False)
    registered_name = Column(String, nullable=False)
    author = Column(String, default='default', nullable=False)
    name = Column(String, default='default', nullable=False)
    version = Column(Integer, nullable=False)
    version_description = Column(String, default='')

    # Persistence of fitted states
    has_external_files = Column(Boolean, default=False)
    filepaths = Column(JSONB, default={})

    # Generic store and metadata for all child objects
    metadata_ = Column('metadata', JSONB, default={})


    def __init__(self, name='default', has_external_files=False,
                 author='default', version_description=None, **kwargs):
        # Initialize values expected to exist at time of instantiation
        self.registered_name = self.__class__.__name__
        self.id = uuid.uuid4()
        self.author = author
        self.name = name
        self.has_external_files = has_external_files
        self.version_description = version_description
        self.metadata_ = {}

        # For external loading - initialize to None
        self.unloaded_externals = None

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

        # Store library versions in case of future loads into unsupported environments
        self.metadata_['library_versions'] = INSTALLED_LIBRARIES

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

    def load(self, load_externals=True):
        '''
        Counter operation for save
        Needs to load any file and db objects

        Class definition is stored by registered_name param and
        Pickled objects are stored in external_filename param

        :param load_externals: Boolean flag whether to load the external files
        useful for relationships that only need class definitions and not data
        '''

        # Lookup appropriate class and reinstantiate
        self.__class__ = self._load_class()

        if self.has_external_files and load_externals:
            self._load_external_files()

        if not load_externals:
            self.unloaded_externals=True

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

    def custom_hasher(self, object_to_hash, custom_class_proxy=type(object.__dict__)):
        """
        Adapted from: https://stackoverflow.com/questions/5884066/hashing-a-dictionary
        Makes a hash from a dictionary, list, tuple or set to any level, that
        contains only other hashable types (including any lists, tuples, sets, and
        dictionaries). In the case where other kinds of objects (like classes) need
        to be hashed, pass in a collection of object attributes that are pertinent.
        For example, a class can be hashed in this fashion:

        custom_hasher([cls.__dict__, cls.__name__])

        A function can be hashed like so:

        custom_hasher([fn.__dict__, fn.__code__])
        """
        if type(object_to_hash) == custom_class_proxy:
            o2 = {}
            for k, v in object_to_hash.items():
                if not k.startswith("__"):
                    o2[k] = v
            object_to_hash = o2

        if isinstance(object_to_hash, (set, tuple, list)):
            return tuple([self.custom_hasher(e) for e in object_to_hash])

        elif isinstance(object_to_hash, (pd.DataFrame, pd.Series)):
            return hash_pandas_object(object_to_hash, index=False).sum()

        elif object_to_hash is None:
            # hash of None is unstable between systems
            return -12345678987654321
            
        elif not isinstance(object_to_hash, dict):
            return hash(object_to_hash)

        new_object_to_hash = copy.deepcopy(object_to_hash)
        for k, v in new_object_to_hash.items():
            new_object_to_hash[k] = self.custom_hasher(v)

        return hash(tuple(frozenset(sorted(new_object_to_hash.items()))))
