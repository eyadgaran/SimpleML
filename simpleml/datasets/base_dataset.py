"""
Base Module for Datasets

Two use cases:
    1) Processed, or traditional datasets. In situations of clean,
representative data, this can be used directly for modeling purposes.

    2) Otherwise, a `raw dataset` needs to be created first with a `dataset pipeline`
to transform it into the processed form.
"""

__author__ = "Elisha Yadgaran"


import logging
import weakref
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from simpleml.datasets.dataset_splits import Split
from simpleml.persistables.base_persistable import Persistable
from simpleml.registries import DatasetRegistry
from simpleml.save_patterns.decorators import ExternalArtifactDecorators
from simpleml.utils.errors import DatasetError

if TYPE_CHECKING:
    # Cyclical import hack for type hints
    from simpleml.pipelines.base_pipeline import Pipeline


LOGGER = logging.getLogger(__name__)


@ExternalArtifactDecorators.register_artifact(
    artifact_name='dataset', save_attribute='dataframe', restore_attribute='_external_file')
class Dataset(Persistable, metaclass=DatasetRegistry):
    '''
    Base class for all Dataset objects.

    Every dataset has a "dataframe" object associated with it that is responsible
    for housing the data. The term dataframe is a bit of a misnomer since it
    does not need to be a pandas.DataFrame obejct.

    Each dataframe can be subdivided by inheriting classes and mixins to support
    numerous representations:
    ex: y column for supervised
        train/test/validation splits
        ...

    Datasets can be constructed from scratch or as derivatives of existing datasets.
    In the event of derivation, a pipeline must be specified to transform the
    original data

    -------
    Schema
    -------
    No additional columns
    '''
    object_type: str = 'DATASET'

    def __init__(
        self,
        has_external_files: bool = True,
        label_columns: Optional[List[str]] = None,
        other_named_split_sections: Optional[Dict[str, List[str]]] = None,
        **kwargs,
    ):
        """
        param label_columns: Optional list of column names to register as the "y" split section
        param other_named_split_sections: Optional map of section names to lists of column names for
            other arbitrary split columns -- must match expected consumer signatures (e.g. sample_weights)
            because passed through untouched downstream (eg sklearn.fit(**split))
        All other columns in the dataframe will automatically be referenced as "X"
        """
        # If no save patterns are set, specify a default for disk_pickled
        if 'save_patterns' not in kwargs:
            kwargs['save_patterns'] = {'dataset': ['disk_pickled']}
        super().__init__(
            has_external_files=has_external_files, **kwargs)

        # split sections are an optional set of inputs to register split references
        # for later use. defaults to just `X` and `y` but arbitrary inputs can
        # be passed (eg sample_weights, etc)

        # validate input
        if other_named_split_sections is None:
            other_named_split_sections = {}
        else:
            for k, v in other_named_split_sections.items():
                if not isinstance(v, (list, tuple)):
                    raise DatasetError(
                        f"Split sections must be a map of section reference (eg `y`) to list of columns. {k}: {v} passed instead"
                    )

        self.config["split_section_map"] = {
            # y maps to label columns (by default assume unsupervised so no targets)
            "y": label_columns or [],
            # arbitrary passed others
            **other_named_split_sections
            # everything else automatically becomes "X"
        }

    @property
    def dataframe(self) -> Any:
        """
        Property wrapper to retrieve the external object associated with the
        dataset.
        Automatically checks for unloaded artifacts and loads, if necessary.
        Will attempt to create a new dataframe if external object is not already
        created via `self.build_dataframe()`
        """
        # Return dataframe if generated, otherwise generate first
        self.load_if_unloaded("dataset")

        if not hasattr(self, "_external_file") or self._external_file is None:
            self.build_dataframe()

        return self._dataframe

    @dataframe.setter
    def dataframe(self, df: Any) -> None:
        """
        Exposed setter for the external dataframe object
        Has hooks for data validation that can be customized in inheriting classes
        """
        # run validation
        self._validate_state(df)
        self._validate_data(df)
        self._validate_schema(df)
        self._validate_dtype(df)

        # pass down to actually set attribute
        self._dataframe = df

    @property
    def _dataframe(self) -> Any:
        """
        Separate property method wrapper for the external object
        Allows mixins/subclasses to change behavior of accsessor
        """
        return self._external_file

    @_dataframe.setter
    def _dataframe(self, df: Any) -> None:
        """
        Setter method for self._external_file
        Allows mixins/subclasses to validate input
        """
        self._external_file = df

    def get_section_columns(self, section: str) -> List[str]:
        """
        Helper accessor for column names in the split_section_map
        """
        return self.config.get("split_section_map").get(section, [])

    @property
    def label_columns(self) -> List[str]:
        """
        Keep column list for labels in metadata to persist through saving
        """
        return self.get_section_columns("y")

    def _validate_state(self, df: Any) -> None:
        """
        Hook to validate the persistable state before allowing modification
        """
        # TODO: add orm level restrictions if persistable is already saved
        # can still be circumvented by directly calling low level methods,
        # but shield against naive abuse

    def _validate_data(self, df: Any) -> None:
        """
        Hook to validate the contents of the data
        """

    def _validate_schema(self, df: Any) -> None:
        """
        Hook to validate the schema of the data (columns/sections)
        """

    def _validate_dtype(self, df: Any) -> None:
        """
        Hook to validate the types of the data
        """

    def build_dataframe(self):
        """
        Must set self._external_file
        Cant set as abstractmethod because of database lookup dependency
        """
        raise NotImplementedError

    def add_pipeline(self, pipeline: "Pipeline") -> None:
        """
        Setter method for dataset pipeline used
        """
        self.pipeline = pipeline

    def _hash(self) -> str:
        """
        Datasets rely on external data so instead of hashing only the config,
        hash the actual resulting dataframe
        This requires loading the data before determining duplication
        so overwrite for differing behavior

        Technically there is little reason to hash anything besides the dataframe,
        but a design choice was made to separate the representation of the data
        from the use cases. For example: two dataset objects with different configured
        labels will yield different downstream results, even if the data is identical.
        Similarly with the pipeline, maintaining the back reference to the originating
        source is preferred, even if the final data can be made through different
        transformations.

        Hash is the combination of the:
            1) Dataframe
            2) Config
            3) Pipeline
        """
        dataframe = self.dataframe
        config = self.config
        if self.pipeline is not None:
            pipeline_hash = self.pipeline.hash_ or self.pipeline._hash()
        else:
            pipeline_hash = None

        return self.custom_hasher((dataframe, config, pipeline_hash))

    def save(self, **kwargs) -> None:
        """
        Extend parent function with a few additional save routines
        """
        if self.pipeline is None:
            LOGGER.warning(
                "Not using a dataset pipeline. Correct if this is unintended"
            )

        super().save(**kwargs)

    @property
    def X(self) -> Any:
        """
        Return the subset that isn't in the target labels
        """
        raise NotImplementedError

    @property
    def y(self) -> Any:
        """
        Return the target label columns
        """
        raise NotImplementedError

    def get(self, column: str, split: str) -> Any:
        """
        Unimplemented method to explicitly split X and y
        Must be implemented by subclasses
        """
        raise NotImplementedError

    def get_feature_names(self) -> List[str]:
        """
        Should return a list of the features in the dataset
        """
        raise NotImplementedError

    def get_split(self, split: str) -> Split:
        """
        Uninplemented method to return a Split object

        Differs from the main get method by wrapping with an internal
        interface class (`Split`). Agnostic to implementation library
        and compatible with downstream SimpleML consumers (pipelines, models)
        """
        raise NotImplementedError

    def get_split_names(self) -> List[str]:
        """
        Uninplemented method to return the split names available for the dataset
        """
        raise NotImplementedError
