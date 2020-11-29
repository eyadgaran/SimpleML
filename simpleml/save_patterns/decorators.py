'''
functions and decorators to extend default save patterns
'''

__author__ = 'Elisha Yadgaran'


import logging
from typing import Type, Optional, Callable, Union

from simpleml.utils.errors import SimpleMLError
from simpleml.registries import SAVE_METHOD_REGISTRY, LOAD_METHOD_REGISTRY


LOGGER = logging.getLogger(__name__)


class SavePatternDecorators(object):
    '''
    Decorators that can be used for registering methods for loading
    and saving.
    '''
    @staticmethod
    def register_save_pattern(
        cls_or_save_pattern: Optional[Union[str, Type]] = None,
        save: Optional[bool] = True,
        load: Optional[bool] = True,
        overwrite: Optional[bool] = False
    ) -> Callable:
        '''
        Decorates a class to register the method(s) to use for saving and/or
        loading for the particular pattern

        IT IS ALLOWABLE TO HAVE DIFFERENT CLASSES HANDLE SAVING AND LOADING FOR
        THE SAME REGISTERED PATTERN

        :param cls_or_save_pattern: the optional string or class denoting the pattern this
            class implements (e.g. `disk_pickled`). Checks class attribute
            `cls.SAVE_PATTERN` if null
            cls is automatically passed when calling decorator without parameters
            (@SavePatternDecorators.register_save_pattern)
        :param save: optional bool; default true; whether to use the decorated
            class as the save method for the registered save pattern
        :param load: optional bool; default true; whether to use the decorated
            class as the load method for the registered save pattern
        :param overwrite: optional bool; default false; whether to overwrite the
            the registered class for the save pattern, if it exists. Otherwise throw
            an error
        '''
        if isinstance(cls_or_save_pattern, str):
            cls = None
            save_pattern = cls_or_save_pattern
        else:
            cls = cls_or_save_pattern
            save_pattern = None

        def register(cls: Type) -> Type:
            register_save_pattern(
                cls=cls,
                save_pattern=save_pattern,
                save=save,
                load=load
            )
            return cls

        if cls is None:
            return register
        else:
            return register(cls)

    @staticmethod
    def deregister_save_pattern(
        cls_or_save_pattern: Optional[str] = None,
        save: Optional[bool] = True,
        load: Optional[bool] = True
    ) -> Callable:
        '''
        Class level decorator to deregister allowed save patterns. Doesnt
        actually make use of the class but included for completeness.
        Recommended to use importable `deregister_save_pattern` function directly

        :param cls_or_save_pattern: the optional string or class denoting the pattern this
            class implements (e.g. `disk_pickled`). Checks class attribute
            `cls.SAVE_PATTERN` if null
            cls is automatically passed when calling decorator without parameters
            (@SavePatternDecorators.deregister_save_pattern)
        :param save: optional bool; default true; whether to drop the decorated
            class as the save method for the registered save pattern
        :param load: optional bool; default true; whether to drop the decorated
            class as the load method for the registered save pattern
        '''
        if isinstance(cls_or_save_pattern, str):
            cls = None
            save_pattern = cls_or_save_pattern
        else:
            cls = cls_or_save_pattern
            save_pattern = None

        def deregister(cls: Type) -> Type:
            deregister_save_pattern(
                cls=cls,
                save_pattern=save_pattern,
                save=save,
                load=load
            )
            return cls

        if cls is None:
            return deregister
        else:
            return deregister(cls)


'''
Function form for explicit registration
'''


def register_save_pattern(
    cls: Type,
    save_pattern: Optional[str] = None,
    save: Optional[bool] = True,
    load: Optional[bool] = True,
    overwrite: Optional[bool] = False
) -> None:
    '''
    Register the class to use for saving and
    loading for the particular pattern

    IT IS ALLOWABLE TO HAVE DIFFERENT CLASSES HANDLE SAVING AND LOADING FOR
    THE SAME REGISTERED PATTERN

    :param save_pattern: the optional string denoting the pattern this
        class implements (e.g. `disk_pickled`). Checks class attribute
        `cls.SAVE_PATTERN` if null
    :param save: optional bool; default true; whether to use the decorated
        class as the save method for the registered save pattern
    :param load: optional bool; default true; whether to use the decorated
        class as the load method for the registered save pattern
    :param overwrite: optional bool; default false; whether to overwrite the
        the registered class for the save pattern, if it exists. Otherwise throw
        an error
    '''
    if save_pattern is None:
        if not hasattr(cls, 'SAVE_PATTERN'):
            raise SimpleMLError('Cannot register save pattern without passing the `save_pattern` parameter or setting the class attribute `cls.SAVE_PATTERN`')
        save_pattern = cls.SAVE_PATTERN

    # Independent registration for saving and loading
    if save:
        SAVE_METHOD_REGISTRY.register(save_pattern, cls, allow_duplicates=overwrite)

    if load:
        LOAD_METHOD_REGISTRY.register(save_pattern, cls, allow_duplicates=overwrite)


def deregister_save_pattern(
    cls: Optional[Type] = None,
    save_pattern: Optional[str] = None,
    save: Optional[bool] = True,
    load: Optional[bool] = True
) -> None:
    '''
    Deregister the class to use for saving and
    loading for the particular pattern

    :param save_pattern: the optional string denoting the pattern this
        class implements (e.g. `disk_pickled`). Checks class attribute
        `cls.SAVE_PATTERN` if null
    :param save: optional bool; default true; whether to remove the
        class as the save method for the registered save pattern
    :param load: optional bool; default true; whether to remove the
        class as the load method for the registered save pattern
    '''
    if save_pattern is None:
        if not hasattr(cls, 'SAVE_PATTERN'):
            raise SimpleMLError('Cannot deregister save pattern without passing the `save_pattern` parameter or setting the class attribute `cls.SAVE_PATTERN`')
        save_pattern = cls.SAVE_PATTERN

    # Independent deregistration for saving and loading
    if save and save_pattern in SAVE_METHOD_REGISTRY.registry:
        if cls is not None and SAVE_METHOD_REGISTRY.get(save_pattern) != cls:
            LOGGER.warning(f"Deregistering {save_pattern} as save pattern but passed class does not match registered class")
        SAVE_METHOD_REGISTRY.drop(save_pattern)

    if load and save_pattern in LOAD_METHOD_REGISTRY.registry:
        if cls is not None and LOAD_METHOD_REGISTRY.get(save_pattern) != cls:
            LOGGER.warning(f"Deregistering {save_pattern} as load pattern but passed class does not match registered class")
        LOAD_METHOD_REGISTRY.drop(save_pattern)


class ExternalArtifactDecorators(object):
    '''
    Decorators for artifact de/registration
    Expected to be applied at the class level to add class attributes indicating
    registered artifacts
    '''
    @staticmethod
    def register_artifact(
        artifact_name: str,
        save_attribute: str,
        restore_attribute: str
    ) -> Callable:
        '''
        Class level decorator to define artifacts produced. Expects each class to
        implement as many as needed to accomodate.

        Format:
        ```
        @register_artifact(artifact_name='model', save_attribute='wrapper_attribute', restore_attribute='_internal_attribute')
        class NewPersistable(Persistable):
            @property
            def wrapper_attribute(self):
                if not hasattr(self, _internal_attribute):
                    self._internal_attribute = self.create_attribute()
                return self._internal_attribute
        ```
        Intentionally specify different attributes for saving and restoring
        to allow developer to wrap attribute in property decorator for
        lazy caching
        '''
        def register(cls: Type) -> Type:
            register_artifact(cls, artifact_name, save_attribute, restore_attribute)
            return cls
        return register

    @staticmethod
    def deregister_artifact(artifact_name: str) -> Callable:
        '''
        Class level decorator to deregister artifacts produced. Expects each class to
        implement as many as needed to accomodate.
        Expected to be used by subclasses that redefine artifacts but dont
        want to expose the possibility of a developer accessing them.
        (By default registering artifacts only exposes them to be persisted if
        declared in save_methods)
        '''
        def deregister(cls: Type) -> Type:
            deregister_artifact(cls, artifact_name)
            return cls
        return deregister


'''
Function form for explicit registration
'''


def register_artifact(
    cls: Type,
    artifact_name: str,
    save_attribute: str,
    restore_attribute: str
) -> None:
    '''
    Register the artifact for potential persistence by a save pattern
    '''
    registered_attribute = f'_ARTIFACT_{artifact_name}'
    setattr(cls, registered_attribute, {'save': save_attribute, 'restore': restore_attribute})


def deregister_artifact(
    cls: Type,
    artifact_name: str
) -> None:
    '''
    Deregister the artifact from being able to be persisted for this class
    '''
    registered_attribute = f'_ARTIFACT_{artifact_name}'
    if hasattr(cls, registered_attribute):
        delattr(cls, registered_attribute)
