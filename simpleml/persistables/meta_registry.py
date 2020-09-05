'''
Meta class to auto register new classes
'''
from sqlalchemy.ext.declarative import declarative_base
from abc import ABCMeta
import logging

__author__ = 'Elisha Yadgaran'

LOGGER = logging.getLogger(__name__)


class Registry(object):
    '''
    Importable class to maintain reference to the global registry
    '''

    def __init__(self):
        self.registry = {}

    def register(self, cls):
        # Check for class duplication. Some workflows reload everything and
        # that is ok. As long as the definitions are the same
        if cls.__name__ in self.registry and cls is not self.registry[cls.__name__]:
            raise ValueError('Cannot duplicate class in registry: {}'.format(cls.__name__))
        self.registry[cls.__name__] = cls

    def get_from_registry(self, class_name):
        cls = self.registry.get(class_name)
        if cls is None:
            LOGGER.error('Class not found for {}. Make sure to import the class into the registry before calling'.format(class_name))
        return cls

    def get(self, class_name):
        return self.get_from_registry(class_name)


class NamedRegistry(Registry):
    '''
    Explicitly named version of the registry (not implicit on class names)
    '''

    def register(self, name, cls, allow_duplicates=True):
        # Check for duplication
        if name in self.registry and cls is not self.registry[name]:
            LOGGER.warning(f'Attempting to overwrite class in registry: {name}')
            if not allow_duplicates:
                raise ValueError(f'Cannot overwrite class in registry: {name}')
        self.registry[name] = cls


# Importable registry
# NEED to use consistent import pattern, otherwise will refer to different memory objects
# from meta_register import SIMPLEML_REGISTRY as s1 != from simpleml.persistables.meta_register import SIMPLEML_REGISTRY as s2
SIMPLEML_REGISTRY = Registry()

# Need to explicitly merge metaclasses to avoid conflicts
MetaBase = type(declarative_base())


class MetaRegistry(MetaBase, ABCMeta):
    def __new__(cls, clsname, bases, attrs):
        newclass = super(MetaRegistry, cls).__new__(cls, clsname, bases, attrs)
        SIMPLEML_REGISTRY.register(newclass)
        return newclass

    '''
    TBD on implementing registry as class attribute

    def __init__(cls, name, bases, nmspc):
        super(MetaRegistry, cls).__init__(name, bases, nmspc)

        if not hasattr(cls, 'registry'):
            cls.registry = set()

        cls.registry.add(cls)

        # Remove base classes
        cls.registry -= set(bases)

    def __iter__(cls):
        return iter(cls.registry)
    '''


# Instantiate specific persistable registries for easy lookup of object types
DATASET_REGISTRY = Registry()
PIPELINE_REGISTRY = Registry()
MODEL_REGISTRY = Registry()
METRIC_REGISTRY = Registry()


class DatasetRegistry(MetaRegistry):
    def __new__(cls, clsname, bases, attrs):
        newclass = super(DatasetRegistry, cls).__new__(cls, clsname, bases, attrs)
        DATASET_REGISTRY.register(newclass)
        return newclass


class PipelineRegistry(MetaRegistry):
    def __new__(cls, clsname, bases, attrs):
        newclass = super(PipelineRegistry, cls).__new__(cls, clsname, bases, attrs)
        PIPELINE_REGISTRY.register(newclass)
        return newclass


class ModelRegistry(MetaRegistry):
    def __new__(cls, clsname, bases, attrs):
        newclass = super(ModelRegistry, cls).__new__(cls, clsname, bases, attrs)
        MODEL_REGISTRY.register(newclass)
        return newclass


class MetricRegistry(MetaRegistry):
    def __new__(cls, clsname, bases, attrs):
        newclass = super(MetricRegistry, cls).__new__(cls, clsname, bases, attrs)
        METRIC_REGISTRY.register(newclass)
        return newclass


# Importable registry for all custom keras objects
# Keras has an annoying persistence pattern that only supports native class references
# Custom class objects need to be passed in at load time
KERAS_REGISTRY = Registry()


class KerasRegistry(ABCMeta):
    def __new__(cls, clsname, bases, attrs):
        newclass = super(KerasRegistry, cls).__new__(cls, clsname, bases, attrs)
        KERAS_REGISTRY.register(newclass)
        return newclass
