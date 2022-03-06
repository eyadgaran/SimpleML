"""
Module for base save pattern definition

Starts by serializing to temp folder and formatting
"""

__author__ = "Elisha Yadgaran"


import logging
from typing import Any, Dict, Tuple, Type

LOGGER = logging.getLogger(__name__)


class BaseSavePattern(object):
    """
    Base class for save patterns (registered wrappers for the collection of
    serializers and deserializers)
    '''
    serializers: Tuple[Type['BaseSerializer']] = tuple()
    deserializers: Tuple[Type['BaseSerializer']] = tuple()

    @classmethod
    def save(cls, **kwargs) -> Dict[str, str]:
        """
        Routine to iterate through serializers returning the final metadata
        """
        if not cls.serializers:
            raise ValueError("Need to specify at least one serialization class")
        for serializer in cls.serializers:
            LOGGER.debug(f"Serializing with {serializer}")
            params = serializer.serialize(**kwargs)
            LOGGER.debug(f"Serialization params: {params}")
            kwargs.update(params)
        return params

    @classmethod
    def load(cls, **kwargs) -> Any:
        """
        The load method invoked
        """
        if not cls.deserializers:
            raise ValueError("Need to specify at least one deserialization class")

        for deserializer in cls.deserializers:
            LOGGER.debug(f"Deserializing with {deserializer}")
            params = deserializer.deserialize(**kwargs)
            LOGGER.debug(f"Deserialization params: {params}")
            kwargs.update(params)
        return params["obj"]


class BaseSerializer(object):
    @staticmethod
    def serialize(**kwargs) -> Dict[str, str]:
        raise NotImplementedError

    @staticmethod
    def deserialize(**kwargs) -> Dict[str, Any]:
        raise NotImplementedError
