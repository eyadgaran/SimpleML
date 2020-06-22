from sqlalchemy import Column, func, String, Boolean, Integer
from simpleml.persistables.meta_registry import MetaRegistry, SIMPLEML_REGISTRY
from simpleml.persistables.sqlalchemy_types import GUID, JSON
from simpleml.persistables.base_sqlalchemy import SimplemlCoreSqlalchemy
from simpleml.persistables.saving import AllSaveMixin
from simpleml.persistables.hashing import CustomHasherMixin
from simpleml.utils.library_versions import INSTALLED_LIBRARIES
from simpleml.utils.errors import SimpleMLError
import uuid
from abc import abstractmethod
from future.utils import with_metaclass
from collections import defaultdict
from typing import Dict, Union, Optional, Any
import logging


__author__ = 'Elisha Yadgaran'


LOGGER = logging.getLogger(__name__)


class Persistable(with_metaclass(MetaRegistry, SimplemlCoreSqlalchemy, AllSaveMixin, CustomHasherMixin)):
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
    project: Project objects are associated with. Useful if multiple persistables
        relate to the same project and want to be grouped (but have different names)
        also good for implementing row based security across teams
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
    author = Column(String, default='default', nullable=False)
    project = Column(String, default='default', nullable=False)
    name = Column(String, default='default', nullable=False)
    version = Column(Integer, nullable=False)
    version_description = Column(String, default='')

    # Persistence of fitted states
    has_external_files = Column(Boolean, default=False)
    filepaths = Column(JSON, default={})

    # Generic store and metadata for all child objects
    metadata_ = Column('metadata', JSON, default={})

    # Internal Registry for all allowed external files
    # Does not need to be persisted because it gets populated on import
    # (and can therefore be changed between versions)
    ARTIFACTS = {}

    def __init__(self, name=None, has_external_files=False,
                 author=None, project=None, version_description=None,
                 save_method=None, **kwargs):
        # Initialize values expected to exist at time of instantiation
        self.registered_name = self.__class__.__name__
        self.id = uuid.uuid4()
        self.author = author
        self.project = project
        self.name = name
        self.has_external_files = has_external_files
        self.version_description = version_description

        if has_external_files and save_method is None:
            LOGGER.warn('Persistable has external artifacts, but has not specified any save methods. Defaulting to local `disk_pickled`')
            save_method = defaultdict(['disk_pickled'])

        # Special place for SimpleML internal params
        # Think of as the config to initialize objects
        self.metadata_ = {}  # Place for any arbitrary metadata
        self.metadata_['config'] = {}  # Place for parameters that uniquely configure an instance on initialization
        self.metadata_['state'] = {}  # Place for transitory values that may be set post initialization (and want to be persisted)

        # For external loading - initialize to None
        self.unloaded_artifacts = []
        # Store save method in state metadata as an operational setting, otherwise
        # it could affect the hash and result in a different object per save location
        self.state['save_method'] = save_method

    @property
    def config(self):
        return self.metadata_['config']

    @property
    def state(self):
        return self.metadata_['state']

    @property
    def library_versions(self):
        return self.metadata_.get('library_versions', {})

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

    def save(self):
        '''
        Each subclass needs to instantiate a save routine to persist to the
        database and any other required filestore

        sqlalchemy_mixins supports active record style TableModel.save()
        so can still call super(Persistable, self).save()
        '''
        if self.has_external_files:
            self.save_external_files()

        # Hash contents upon save
        self.hash_ = self._hash()

        # Get the latest version for this "friendly name"
        self.version = self._get_latest_version()

        # Store library versions in case of future loads into unsupported environments
        self.metadata_['library_versions'] = INSTALLED_LIBRARIES

        super(Persistable, self).save()

    def save_external_files(self):
        '''
        Main routine to save registered external artifacts. Each save pattern
        is defined using the standard api for the save params defined here. If
        a pattern requires more imports, it needs to be added here

        Uses a standardized nomenclature to reuse params regardless of save method
        {
            'persistable_id': the database id of the persistable. typically used as the root name of the saved object. implementations will pre/suffix,
            'persistable_type': the persistable type (DATASET/PIPELINE..),
            'overwrite': boolean. shortcut in case save method redefines a serialization routine
        }
        '''
        save_params: Dict[str, Union[str, bool]]
        save_params = {
            'persistable_id': str(self.id),
            'persistable_type': self.object_type,
            'overwrite': False,
        }
        # Iterate through each artifact and save
        for artifact_name, save_methods in self.state.get('save_methods', {}).items():
            # Artifact has to be registered in self.ARTIFACTS
            obj = self.get_artifact(artifact_name)
            # Iterate through list of save methods
            for save_method in save_methods:
                self.save_external_file(obj=obj, **save_params)

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

        # Track the list of artifacts
        # New persistables without a specified filepath dictionary have type
        # sqlalchemy.sql.schema.Column - calling list(Column.keys()) would fail
        if not isinstance(self.filepaths, dict):
            LOGGER.warning('Load appears to being called on an unsaved Persistable')
            self.unloaded_artifacts = []
        else:
            self.unloaded_artifacts = list(self.filepaths.keys())

        if self.has_external_files and load_externals:
            self.load_external_files()

    def load_external_files(self, artifact_name: Optional[str] = None):
        '''
        Main routine to restore registered external artifacts. Will iterate
        through save patterns and break after the first successful restore
        (allows robustness in the event of unavailable resources)
        '''
        def _load(artifact_name: str, save_methods: Dict[str, Any]):
            # Iterate through dict of save methods and file data
            for save_method in save_methods:
                try:
                    obj = self.load_external_file(artifact_name, save_method)
                    self.restore_artifact(artifact_name, obj)
                    break
                except Exception as e:
                    LOGGER.error(f'Failed to restore {artifact_name} via {save_method} ({e}). Trying next save pattern...')
            else:
                raise SimpleMLError(f'Unable to restore {artifact_name} via any registered pattern')

        # Iterate through each artifact and restore
        if artifact_name is None:
            for artifact_name, save_methods in self.filepaths.items():
                _load(artifact_name, save_methods)
        else:
            _load(artifact_name, self.filepaths.get(artifact_name, {}))

    def load_if_unloaded(self, artifact_name: str) -> None:
        '''
        Convenience method to load an artifact if not already loaded.
        Easy dropin in property methods
        ```
        @property
        def artifact(self):
            self.load_if_unloaded(artifact_name)
            if not hasattr(self, artifact_attribute):
                self.create_artifact()
            return self.artifact_attribute
        ```
        '''
        if artifact_name in self.unloaded_artifacts:
            self.load_external_files(artifact_name=artifact_name)

    def _load_class(self):
        '''
        Wrapper function to call global registry of all imported class names
        '''
        return SIMPLEML_REGISTRY.get(self.registered_name)
