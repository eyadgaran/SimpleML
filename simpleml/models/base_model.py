"""
Base Model module
"""

__author__ = "Elisha Yadgaran"

import logging
import uuid
import weakref
from abc import abstractmethod
from typing import Any, Dict, Optional, Union

import numpy as np

from simpleml.persistables.base_persistable import Persistable
from simpleml.pipelines import Pipeline
from simpleml.registries import ModelRegistry
from simpleml.save_patterns.decorators import ExternalArtifactDecorators
from simpleml.utils.errors import ModelError

LOGGER = logging.getLogger(__name__)


@ExternalArtifactDecorators.register_artifact(
    artifact_name="model",
    save_attribute="external_model",
    restore_attribute="_external_file",
)
class Model(Persistable, metaclass=ModelRegistry):
    """
    Base class for all Model objects. Defines the required
    parameters for versioning and all other metadata can be
    stored in the arbitrary metadata field

    Also outlines the expected subclass methods (with NotImplementedError).
    Design choice to not abstract unified API across all libraries since each
    has a different internal mechanism
    """

    object_type = "MODEL"

    def __init__(
        self,
        has_external_files: bool = True,
        external_model_kwargs: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        fitted: bool = False,
        pipeline_id: Optional[Union[str, uuid.uuid4]] = None,
        **kwargs,
    ):
        """
        Need to explicitly separate passthrough kwargs to external models since
        most do not support arbitrary **kwargs in the constructors

        Two supported patterns - full initialization in constructor or stepwise configured
        before fit and save
        """
        # If no save patterns are set, specify a default for disk_pickled
        if "save_patterns" not in kwargs:
            kwargs["save_patterns"] = {"model": ["disk_pickled"]}
        super(Model, self).__init__(has_external_files=has_external_files, **kwargs)

        # Prep params for model instantiation. lazy init (first reference will create it)
        if external_model_kwargs is None:
            external_model_kwargs = {}
        self._external_model_init_kwargs = external_model_kwargs
        if params is not None:
            self.set_params(**params)

        # Initialize as unfitted
        self.fitted = fitted
        # initialize null pipeline reference
        self.pipeline_id = pipeline_id

    @property
    def fitted(self):
        return self.state.get("fitted")

    @fitted.setter
    def fitted(self, value):
        self.state["fitted"] = value

    @property
    def external_model(self):
        """
        All model objects are going to require some filebase persisted object

        Wrapper around whatever underlying class is desired
        (eg sklearn or keras)
        """
        self.load_if_unloaded("model")

        # lazy build
        if not hasattr(self, "_external_file"):
            self._external_file = self._create_external_model(
                **self._external_model_init_kwargs
            )
            # clear temp vars
            del self._external_model_init_kwargs

        return self._external_file

    def _create_external_model(self, **kwargs):
        """
        Abstract method for each subclass to implement

        should return the desired model object
        """
        raise NotImplementedError

    def add_pipeline(self, pipeline: Pipeline) -> None:
        """
        Setter method for dataset pipeline used
        """
        if pipeline is None:
            return
        self.pipeline_id = pipeline.id
        self.pipeline = pipeline

    @property
    def pipeline(self):
        """
        Use a weakref to bind linked pipeline so it doesnt bloat usage
        returns pipeline if still available or tries to fetch otherwise
        """
        # still referenced weakref
        if hasattr(self, "_pipeline") and self._pipeline() is not None:
            return self._pipeline()

        # null return if no associated pipeline (governed by pipeline_id)
        elif self.pipeline_id is None:
            return None

        # else regenerate weakref
        LOGGER.info("No referenced object available. Refreshing weakref")
        pipeline = self._load_pipeline()
        self._pipeline = weakref.ref(pipeline)
        return pipeline

    @pipeline.setter
    def pipeline(self, pipeline: Pipeline) -> None:
        """
        Need to be careful not to set as the orm pipeline
        Cannot load if wrong type because of recursive behavior (will
        propagate down the whole dependency chain)
        """
        self._pipeline = weakref.ref(pipeline)

    def _load_pipeline(self):
        """
        Helper to fetch the pipeline
        """
        return self.orm_cls.load_pipeline(self.pipeline_id)

    def assert_pipeline(self, msg=""):
        """
        Helper method to raise an error if pipeline isn't present and configured
        """
        if self.pipeline is None or not self.pipeline.fitted:
            raise ModelError(msg)

    def assert_fitted(self, msg=""):
        """
        Helper method to raise an error if model isn't fit
        """
        if not self.fitted:
            raise ModelError(msg)

    def _hash(self):
        """
        Hash is the combination of the:
            1) Pipeline
            2) Model
            3) Params
            4) Config
        May only include attributes that exist at instantiation.
        Any attribute that gets calculated later will result in a race condition
        that may return a different hash depending on when the function is called
        """
        pipeline_hash = self.pipeline.hash_ or self.pipeline._hash()
        model = self.external_model.__class__.__name__
        params = self.get_params()
        config = self.config

        return self.custom_hasher((pipeline_hash, model, params, config))

    def save(self, **kwargs):
        """
        Extend parent function with a few additional save routines

        1) save params
        2) save feature metadata
        """
        self.assert_pipeline("Must set pipeline before saving")
        self.assert_fitted("Must fit model before saving")

        # log only attributes - can be refreshed on each save (does not take effect on reloading)
        self.params = self.get_params(**kwargs)
        self.feature_metadata = self.get_feature_metadata(**kwargs)

        super(Model, self).save(**kwargs)

    def fit(self, **kwargs):
        """
        Pass through method to external model after running through pipeline
        """
        self.assert_pipeline("Must set pipeline before fitting")

        if self.fitted:
            LOGGER.warning("Cannot refit model, skipping operation")
            return self

        if kwargs:
            LOGGER.warning(
                "Attempting to pass runtime parameters to fit. All parameters must be initialized with the constructor - Ignoring input!"
            )

        # Call actual library version fit routine (without passed parameters)
        self._fit()

        # Mark the state so it doesnt get refit and can now be saved
        self.fitted = True

        return self

    def _fit(self):
        """
        Abstract method to act as a placeholder. Inheriting classes MUST instantiate
        this method to manage the fit operation. Intentionally not abstracting
        function because each library internally configures a little differently
        """
        raise NotImplementedError

    def transform(self, *args, **kwargs):
        """
        Run input through pipeline -- only method that should reference the pipeline
        relationship directly (gates the connection point for easy extension in the future)
        """
        return self.pipeline.transform(*args, **kwargs)

    def _predict(self, X, **kwargs):
        """
        Separate out actual predict call for optional overwrite in subclasses
        """
        return self.process(self.external_model.predict, X)

    def predict(self, X, transform=True, **kwargs):
        """
        Pass through method to external model after running through pipeline
        :param transform: bool, whether to transform input via pipeline
         before predicting, default True
        """
        self.assert_fitted("Must fit model before predicting")

        if transform:
            # Pipeline returns Split object if input is null
            # Otherwise transformed matrix
            transformed = self.transform(X, **kwargs)
            X = transformed.X if X is None else transformed

        if (
            X is None
        ):  # Don't attempt to run through model if no samples (can't evaulate ahead of transform in case dataset split used)
            return np.array([])

        return self._predict(X, **kwargs)

    def fit_predict(self, **kwargs):
        """
        Wrapper for fit and predict methods
        """
        self.fit()
        # Pass X as none to cascade using internal dataset for X
        # Assumes only applies to default (training) split
        return self.predict(X=None, **kwargs)

    def get_labels(self, dataset_split=None):
        """
        Wrapper method to return labels from dataset
        """
        return self.pipeline.y(split=dataset_split)

    """
    Pass-through methods to external model
    """

    def get_params(self, **kwargs):
        """
        Pass through method to external model
        """
        return self.external_model.get_params(**kwargs)

    def set_params(self, **params):
        """
        Pass through method to external model
        """
        return self.external_model.set_params(**params)

    def score(self, X, y=None, **kwargs):
        """
        Pass through method to external model
        """
        return self.external_model.score(X, y, **kwargs)

    def get_feature_metadata(self, **kwargs):
        """
        Abstract method for each model to define

        Should return a dict of feature information (importance, coefficients...)
        """
        return self.external_model.get_feature_metadata(
            features=self.pipeline.get_feature_names(), **kwargs
        )


class LibraryModel(Model):
    """
    Main model class needs to be initialize-able in order to play nice with
    database persistence and loading. This class is the in between that defines
    the expected methods for each extended library.

    Examples:
    Scikit-learn estimators --> SklearnModel(LibraryModel): ...
    Keras estimators --> KerasModel(LibraryModel): ...
    PyTorch ...
    ...
    """

    @abstractmethod
    def _fit(self):
        pass
