'''
Meta class to auto register new classes
'''
from sqlalchemy.ext.declarative import declarative_base

__author__ = 'Elisha Yadgaran'


class Registry(object):
    '''
    Importable class to maintain reference to the correct global registry

    (Import splintering makes it non trivial to import registry explicitly)
    '''
    def __init__(self):
        self.registry = {}

    def register(self, cls):
        if cls.__name__ in self.registry:
            raise ValueError('Cannot duplicate class in registry: {}'.format(cls.__name__))
        self.registry[cls.__name__] = cls

    def get_from_registry(self, class_name):
        return self.registry.get(class_name)

    def get(self, class_name):
        return self.get_from_registry(class_name)


# Importable registry
# NEED to use consistent import pattern, otherwise will refer to different memory objects
# from meta_register import SIMPLEML_REGISTRY as s1 != from simpleml.persistables.meta_register import SIMPLEML_REGISTRY as s2
SIMPLEML_REGISTRY = Registry()

# Need to explicitly merge metaclasses to avoid conflicts
MetaBase = type(declarative_base())


class MetaRegistry(MetaBase):
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
