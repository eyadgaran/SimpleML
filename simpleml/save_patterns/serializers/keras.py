"""
Module for Keras save patterns
"""

__author__ = "Elisha Yadgaran"


from os.path import isdir, isfile, join
from typing import Any, Dict

from simpleml.imports import load_model
from simpleml.registries import FILEPATH_REGISTRY, KERAS_REGISTRY
from simpleml.save_patterns.base import BaseSerializer
from simpleml.utils.configuration import (
    HDF5_DIRECTORY,
    TENSORFLOW_SAVED_MODEL_DIRECTORY,
)


class KerasPersistenceMethods(object):
    """
    Base class for internal Keras serialization/deserialization options
    """

    @staticmethod
    def save_model(model: Any, filepath: str, overwrite: bool = True, **kwargs) -> None:
        """
        Serializes an object to the filesystem in Keras native format.

        :param overwrite: Boolean indicating whether to first check if
            object is already serialized. Defaults to not checking, but can be
            leverage by implementations that want the same artifact in multiple
            places
        """
        if not overwrite:
            # Check if file/folder was already serialized
            if isfile(filepath) or isdir(filepath):
                return
        model.save(filepath, **kwargs)

    @staticmethod
    def load_model(filepath: str, **kwargs) -> Any:
        """
        Loads a Keras object from the filesystem.
        """
        return load_model(filepath, custom_objects=KERAS_REGISTRY.registry, **kwargs)

    @staticmethod
    def save_weights(
        model: Any, filepath: str, overwrite: bool = True, **kwargs
    ) -> None:
        """
        Serializes an object to the filesystem in Keras native format.

        :param overwrite: Boolean indicating whether to first check if
            object is already serialized. Defaults to not checking, but can be
            leverage by implementations that want the same artifact in multiple
            places
        """
        if not overwrite:
            # Check if file/folder was already serialized
            if isfile(filepath) or isdir(filepath):
                return
        model.save_weights(filepath, **kwargs)

    @staticmethod
    def load_weights(model: Any, filepath: str, **kwargs) -> Any:
        """
        Loads a Keras object from the filesystem.
        """
        load_status = model.load_weights(filepath, **kwargs)

        # `assert_consumed` can be used as validation that all variable values have been
        # restored from the checkpoint. See `tf.train.Checkpoint.restore` for other
        # methods in the Status object.
        load_status.assert_consumed()

        return model


"""
See https://www.tensorflow.org/guide/keras/save_and_serialize for serialization
options

Whole-model saving & loading
You can save an entire model to a single artifact. It will include:

The model's architecture/config
The model's weight values (which were learned during training)
The model's compilation information (if compile() was called)
The optimizer and its state, if any (this enables you to restart training where you left)
APIs
model.save() or tf.keras.models.save_model()
tf.keras.models.load_model()
There are two formats you can use to save an entire model to disk: the TensorFlow SavedModel format, and the older Keras H5 format. The recommended format is SavedModel. It is the default when you use model.save().

You can switch to the H5 format by:

Passing save_format='h5' to save().
Passing a filename that ends in .h5 or .keras to save().
SavedModel format
SavedModel is the more comprehensive save format that saves the model architecture, weights, and the traced Tensorflow subgraphs of the call functions. This enables Keras to restore both built-in layers as well as custom objects.

Calling model.save('my_model') creates a folder named my_model, containing the following:

Keras H5 format
Keras also supports saving a single HDF5 file containing the model's architecture, weights values, and compile() information. It is a light-weight alternative to SavedModel.

APIs for saving weights to disk & loading them back
Weights can be saved to disk by calling model.save_weights in the following formats:

TensorFlow Checkpoint
HDF5
The default format for model.save_weights is TensorFlow checkpoint. There are two ways to specify the save format:

save_format argument: Set the value to save_format="tf" or save_format="h5".
path argument: If the path ends with .h5 or .hdf5, then the HDF5 format is used. Other suffixes will result in a TensorFlow checkpoint unless save_format is set.
There is also an option of retrieving weights as in-memory numpy arrays. Each API has its pros and cons which are detailed below.


"""


class KerasSavedModelSerializer(BaseSerializer):
    """
    Uses Tensorflow SavedModel serialization

    Output is a folder with `assets  keras_metadata.pb  saved_model.pb  variables`
    """

    @staticmethod
    def serialize(
        obj: Any,
        filepath: str,
        format_directory: str = TENSORFLOW_SAVED_MODEL_DIRECTORY,
        format_extension: str = ".savedModel",
        destination_directory: str = "system_temp",
        **kwargs,
    ) -> Dict[str, str]:

        # Append the filepath to the storage directory
        filepath = join(format_directory, filepath + format_extension)
        full_path = join(FILEPATH_REGISTRY.get(destination_directory), filepath)
        KerasPersistenceMethods.save_model(obj, full_path)
        return {"filepath": filepath, "source_directory": destination_directory}

    @staticmethod
    def deserialize(
        filepath: str, source_directory: str = "system_temp", **kwargs
    ) -> Dict[str, Any]:
        full_path = join(FILEPATH_REGISTRY.get(source_directory), filepath)
        return {"obj": KerasPersistenceMethods.load_model(full_path)}


class KerasH5Serializer(BaseSerializer):
    """
    Uses Keras H5 serialization (legacy behavior)

    Output is a single file
    """

    @staticmethod
    def serialize(
        obj: Any,
        filepath: str,
        format_directory: str = HDF5_DIRECTORY,
        format_extension: str = ".h5",
        destination_directory: str = "system_temp",
        **kwargs,
    ) -> Dict[str, str]:

        # Append the filepath to the storage directory
        filepath = join(format_directory, filepath + format_extension)
        full_path = join(FILEPATH_REGISTRY.get(destination_directory), filepath)
        KerasPersistenceMethods.save_model(obj, full_path, save_format="h5")
        return {"filepath": filepath, "source_directory": destination_directory}

    @staticmethod
    def deserialize(
        filepath: str, source_directory: str = "system_temp", **kwargs
    ) -> Dict[str, Any]:
        full_path = join(FILEPATH_REGISTRY.get(source_directory), filepath)
        return {"obj": KerasPersistenceMethods.load_model(full_path)}
