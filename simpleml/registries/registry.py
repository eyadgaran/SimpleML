'''
Different registries implementations available
'''

__author__ = 'Elisha Yadgaran'


import logging

from typing import Type, Dict

LOGGER = logging.getLogger(__name__)


class Registry(object):
    '''
    Importable class to maintain reference to the global registry
    '''

    def __init__(self):
        self.registry: Dict[str, Type] = {}

    def register(self, cls: Type) -> None:
        # Check for class duplication. Some workflows reload everything and
        # that is ok. As long as the definitions are the same
        if cls.__name__ in self.registry and cls is not self.registry[cls.__name__]:
            raise ValueError('Cannot duplicate class in registry: {}'.format(cls.__name__))
        self.registry[cls.__name__] = cls

    def get_from_registry(self, class_name: str) -> Type:
        cls = self.registry.get(class_name)
        if cls is None:
            LOGGER.error('Class not found for {}. Make sure to import the class into the registry before calling'.format(class_name))
        return cls

    def get(self, class_name: str) -> Type:
        return self.get_from_registry(class_name)

    def drop(self, key: str) -> None:
        '''
        Drop key from registry
        '''
        self.registry.pop(key, None)

    def clear(self) -> None:
        '''
        Clear registry
        '''
        self.registry = {}


class NamedRegistry(Registry):
    '''
    Explicitly named version of the registry (not implicit on class names)
    '''

    def register(self, name: str, cls: Type, allow_duplicates: bool = True) -> None:
        # Check for duplication
        if name in self.registry and cls is not self.registry[name]:
            LOGGER.warning(f'Attempting to overwrite class in registry: {name}')
            if not allow_duplicates:
                raise ValueError(f'Cannot overwrite class in registry: {name}')
        self.registry[name] = cls
