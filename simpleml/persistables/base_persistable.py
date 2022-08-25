"""
Base class for all database tracked records, called "Persistables"
"""

__author__ = "Elisha Yadgaran"


import logging
import uuid
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional, Type, Union

from simpleml.executors.processor import ExecutionProcessor
from simpleml.persistables.hashing import CustomHasherMixin
from simpleml.registries import (
    LOAD_METHOD_REGISTRY,
    ORM_REGISTRY,
    SAVE_METHOD_REGISTRY,
    SIMPLEML_REGISTRY,
    PersistableRegistry,
)
from simpleml.utils.errors import SimpleMLError
from simpleml.utils.library_versions import INSTALLED_LIBRARIES

LOGGER = logging.getLogger(__name__)


class Persistable(CustomHasherMixin, ExecutionProcessor, metaclass=PersistableRegistry):
    """
    Base class for all SimpleML persistable objects.

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
    """

    object_type = "PERSISTABLE"

    def __init__(
        self,
        id: uuid.UUID = None,
        hash_: str = None,
        name: Optional[str] = "default",
        has_external_files: bool = False,
        author: Optional[str] = "default",
        project: Optional[str] = "default",
        version: Optional[int] = None,
        version_description: Optional[str] = "",
        save_patterns: Optional[Dict[str, List[str]]] = None,
        filepaths: Optional[Dict] = None,
        metadata_: Optional[Dict] = None,
        **kwargs,
    ):
        # Initialize values expected to exist at time of instantiation
        self.registered_name: str = self.__class__.__name__
        self.id: uuid.UUID = id or uuid.uuid4()
        self.author = author
        self.project = project
        self.name = name
        self.has_external_files = has_external_files
        self.version_description = version_description

        # expected null values
        self.hash_ = hash_
        self.version = version
        self.filepaths = filepaths or {}

        # Special place for SimpleML internal params
        # Think of as the config to initialize objects
        self.metadata_: Dict[str, Any] = (
            metadata_ or {}
        )  # Place for any arbitrary metadata

        if "config" not in self.metadata_:
            self.metadata_[
                "config"
            ] = (
                {}
            )  # Place for parameters that uniquely configure an instance on initialization
        if "state" not in self.metadata_:
            self.metadata_[
                "state"
            ] = (
                {}
            )  # Place for transitory values that may be set post initialization (and want to be persisted)

        save_patterns = save_patterns or self.state.get("save_patterns")
        if has_external_files and save_patterns is None:
            raise SimpleMLError(
                "Persistable has external artifacts, but has not specified any save patterns.\nTry reinitializing persistable with `Persistable(save_patterns={artifact_name: [save_patterns]})`"
            )

        # Store save pattern in state metadata as an operational setting, otherwise
        # it could affect the hash and result in a different object per save location
        self.state["save_patterns"] = save_patterns

    def __post_init__(self):
        self._configure_unmapped_attributes()

    def __post_restore__(self):
        self._configure_unmapped_attributes()

    def _configure_unmapped_attributes(self):
        """
        Unified entry for unmapped attributes. need to be restored when loading
        classes
        """
        self.unloaded_artifacts: List[str]
        # Track the list of artifacts
        # New persistables without a specified filepath dictionary have type
        # sqlalchemy.sql.schema.Column - calling list(Column.keys()) would fail
        if not isinstance(self.filepaths, dict):
            LOGGER.warning("Load appears to being called on an unsaved Persistable")
            self.unloaded_artifacts = []
        else:
            self.unloaded_artifacts = list(self.filepaths.keys())

    """
    metadata accessor properties
    """

    @property
    def config(self) -> Dict[str, Any]:
        return self.metadata_["config"]

    @property
    def state(self) -> Dict[str, Any]:
        return self.metadata_["state"]

    @property
    def library_versions(self) -> Dict[str, str]:
        return self.metadata_.get("library_versions", {})

    """
    abstract definitions
    """

    @abstractmethod
    def _hash(self):
        """
        Each subclass should implement a hashing routine to uniquely AND consistently
        identify the object contents. Consistency is important to ensure ability
        to assert identity across code definitions
        """

    """
    persistence hooks
    """

    def _get_latest_version(self) -> int:
        """
        Versions should be autoincrementing for each object (constrained over
        friendly name). Executes a database lookup and increments..
        """
        return self.orm_cls.get_latest_version(name=self.name)

    def save(self) -> None:
        """
        Each subclass needs to instantiate a save routine to persist to the
        database and any other required filestore

        sqlalchemy_mixins supports active record style TableModel.save()
        so can still call super(Persistable, self).save()
        """
        if self.has_external_files:  # todo: insert hook to not save on updates
            self.save_external_files()

        # Hash contents upon save
        if self.hash_ is None:
            self.hash_ = self._hash()

        # Get the latest version for this "friendly name"
        if self.version is None:
            self.version = self._get_latest_version()

        # Store library versions in case of future loads into unsupported environments
        if "library_versions" not in self.metadata_:
            self.metadata_["library_versions"] = INSTALLED_LIBRARIES

        self.orm_cls.save_record(**self.to_dict())

    @property
    def orm_cls(self):
        cls = ORM_REGISTRY.get(self.object_type)
        if cls is None:
            raise SimpleMLError(f"No registered ORM class for {self.object_type}")
        return cls

    def save_external_files(self) -> None:
        """
        Main routine to save registered external artifacts. Each save pattern
        is defined using the standard api for the save params defined here. If
        a pattern requires more imports, it needs to be added here

        Uses a standardized nomenclature to reuse params regardless of save pattern
        {
            'persistable_id': the database id of the persistable. typically used as the root name of the saved object. implementations will pre/suffix,
            'persistable_type': the persistable type (DATASET/PIPELINE..),
            'overwrite': boolean. shortcut in case save pattern redefines a serialization routine
        }
        """
        save_params: Dict[str, Union[str, bool]]
        save_params = {
            "persistable_id": str(self.id),
            "persistable_type": self.object_type,
            "overwrite": False,
        }
        # Iterate through each artifact and save
        for artifact_name, save_patterns in self.state.get("save_patterns", {}).items():
            # Artifact has to be registered in self.ARTIFACTS
            obj = self.get_artifact(artifact_name)
            # Iterate through list of save methods
            for save_pattern in save_patterns:
                self.save_external_file(
                    artifact_name=artifact_name,
                    filepath=f"{save_params['persistable_type']}-{save_params['persistable_id']}-{artifact_name}",
                    save_pattern=save_pattern,
                    obj=obj,
                    **save_params,
                )

    def save_external_file(
        self,
        artifact_name: str,
        save_pattern: str,
        cls: Optional[Type] = None,
        **save_params,
    ) -> None:
        """
        Abstracted pattern to save an artifact via one of the registered
        patterns and update the filepaths location
        """
        if cls is None:
            # Look up in registry
            save_cls = SAVE_METHOD_REGISTRY.get(save_pattern)
        else:
            LOGGER.info("Custom save class passed, skipping registry lookup")
            save_cls = cls

        if save_cls is None:
            raise SimpleMLError(f"No registered save pattern for {save_pattern}")

        filepath_data = save_cls.save(artifact_name=artifact_name, **save_params)

        # Update filepaths
        if self.filepaths is None:
            self.filepaths = {}
        if self.filepaths.get(artifact_name, None) is None:
            self.filepaths[artifact_name] = {}
        self.filepaths[artifact_name][save_pattern] = filepath_data

    def get_artifact(self, artifact_name: str) -> Any:
        """
        Accessor method to lookup the artifact in the registry and return
        the corresponding data value
        """
        registered_attribute = f"_ARTIFACT_{artifact_name}"
        if not hasattr(self, registered_attribute):
            raise SimpleMLError(
                "Cannot retrieve artifacts before registering. Make sure to decorate class with @ExternalArtifactDecorators.register_artifact"
            )
        save_attribute = getattr(self, registered_attribute)["save"]
        return getattr(self, save_attribute)

    @classmethod
    def from_dict(cls, **kwargs) -> "Persistable":
        """
        Parameterize a persistable from a dict. Used in deserialization from ORM
        objects
        """
        # skip init calls
        obj = cls.__new__(cls)
        for attr, value in kwargs.items():
            setattr(obj, attr, value)
        # execute __post_restore__ hook
        obj.__post_restore__()
        return obj

    def to_dict(self):
        return vars(self)

    def load_external_files(self, artifact_name: Optional[str] = None) -> None:
        """
        Main routine to restore registered external artifacts. Will iterate
        through save patterns and break after the first successful restore
        (allows robustness in the event of unavailable resources)
        """

        def _load(artifact_name: str, save_patterns: Dict[str, Any]):
            # Iterate through dict of save methods and file data
            for save_pattern in save_patterns:
                try:
                    obj = self.load_external_file(artifact_name, save_pattern)
                    self.restore_artifact(artifact_name, obj)
                    break
                except Exception as e:
                    LOGGER.error(
                        f"Failed to restore {artifact_name} via {save_pattern} ({e}). Trying next save pattern..."
                    )
            else:
                raise SimpleMLError(
                    f"Unable to restore {artifact_name} via any registered pattern"
                )

        # Iterate through each artifact and restore
        # Dont use self.unloaded_artifacts list to force a full reload
        if artifact_name is None:
            for artifact_name, save_patterns in self.filepaths.items():
                _load(artifact_name, save_patterns)
        else:
            _load(artifact_name, self.filepaths.get(artifact_name, {}))

    def load_external_file(
        self, artifact_name: str, save_pattern: str, cls: Optional[Type] = None
    ) -> Any:
        """
        Define pattern for loading external files
        returns the object for assignment
        Inverted operation from saving. Registered functions should take in
        the same data (in the same form) of what is saved in the filepath
        """
        if cls is None:
            # Look up in registry
            load_cls = LOAD_METHOD_REGISTRY.get(save_pattern)
        else:
            LOGGER.info("Custom load class passed, skipping registry lookup")
            load_cls = cls

        if load_cls is None:
            raise SimpleMLError(f"No registered load class for {save_pattern}")

        # Do some validation in case attempting to load unsaved artifact
        artifact = self.filepaths.get(artifact_name, None)
        if artifact is None:
            raise SimpleMLError(f"No artifact saved for {artifact_name}")
        if save_pattern not in artifact:
            raise SimpleMLError(
                f"No artifact saved using save pattern {save_pattern} for {artifact_name}"
            )

        filepath_data = artifact[save_pattern]
        if not isinstance(filepath_data, dict):
            # legacy wrap for old filepath formats
            filepath_data = {"legacy": filepath_data}
        return load_cls.load(**filepath_data)

    def restore_artifact(self, artifact_name: str, obj: Any) -> None:
        """
        Setter method to lookup the restore attribute and set to the passed object
        """
        registered_attribute = f"_ARTIFACT_{artifact_name}"
        if not hasattr(self, registered_attribute):
            raise SimpleMLError(
                "Cannot restore artifacts before registering. Make sure to decorate class with @ExternalArtifactDecorators.register_artifact"
            )
        restore_attribute = getattr(self, registered_attribute)["restore"]
        setattr(self, restore_attribute, obj)

        # Make note that the artifact was loaded
        if hasattr(self, "unloaded_artifacts"):
            try:
                self.unloaded_artifacts.remove(artifact_name)
            except ValueError:
                pass

    def load_if_unloaded(self, artifact_name: str) -> None:
        """
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
        """
        if artifact_name in self.unloaded_artifacts:
            self.load_external_files(artifact_name=artifact_name)
