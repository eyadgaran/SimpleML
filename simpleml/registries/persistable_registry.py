"""
Meta class to auto register new classes with sqlalchemy bases
"""

__author__ = "Elisha Yadgaran"


from abc import ABCMeta

from simpleml.registries.registry import Registry

# Importable registry
# Instantiate specific persistable registries for easy lookup of object types
# NEED to use consistent import pattern, otherwise will refer to different memory objects
# from meta_register import SIMPLEML_REGISTRY as s1 != from simpleml.persistables.meta_register import SIMPLEML_REGISTRY as s2
SIMPLEML_REGISTRY = Registry()
DATASET_REGISTRY = Registry()
PIPELINE_REGISTRY = Registry()
MODEL_REGISTRY = Registry()
METRIC_REGISTRY = Registry()


class PersistableRegistry(ABCMeta):
    """
    Meta class to register SimpleML persistables. expected to be set as
    metaclass for all persistable types
    """

    def __new__(cls, clsname, bases, attrs):
        """
        Metaclass implementation. Called on import of referenced subclasses
        (not called on construction of classes)
        """
        newclass = super(PersistableRegistry, cls).__new__(cls, clsname, bases, attrs)
        SIMPLEML_REGISTRY.register(newclass)
        return newclass

    def __call__(self, *args, **kwargs):
        """
        Overwrite constructor call to add post init hook
        (called when constructing referenced subclasses)
        """
        cls = super().__call__(*args, **kwargs)
        if hasattr(cls, "__post_init__"):
            cls.__post_init__()
        return cls

    """
    TBD on implementing registry as class attribute

    def __init__(cls, name, bases, nmspc):
        super(PersistableRegistry, cls).__init__(name, bases, nmspc)

        if not hasattr(cls, 'registry'):
            cls.registry = set()

        cls.registry.add(cls)

        # Remove base classes
        cls.registry -= set(bases)

    def __iter__(cls):
        return iter(cls.registry)
    """


class DatasetRegistry(PersistableRegistry):
    def __new__(cls, clsname, bases, attrs):
        newclass = super(DatasetRegistry, cls).__new__(cls, clsname, bases, attrs)
        DATASET_REGISTRY.register(newclass)
        return newclass


class PipelineRegistry(PersistableRegistry):
    def __new__(cls, clsname, bases, attrs):
        newclass = super(PipelineRegistry, cls).__new__(cls, clsname, bases, attrs)
        PIPELINE_REGISTRY.register(newclass)
        return newclass


class ModelRegistry(PersistableRegistry):
    def __new__(cls, clsname, bases, attrs):
        newclass = super(ModelRegistry, cls).__new__(cls, clsname, bases, attrs)
        MODEL_REGISTRY.register(newclass)
        return newclass


class MetricRegistry(PersistableRegistry):
    def __new__(cls, clsname, bases, attrs):
        newclass = super(MetricRegistry, cls).__new__(cls, clsname, bases, attrs)
        METRIC_REGISTRY.register(newclass)
        return newclass
