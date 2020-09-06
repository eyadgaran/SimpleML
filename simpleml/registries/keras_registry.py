'''
Keras specific registry. Autoregisters when used as a base class
'''

__author__ = 'Elisha Yadgaran'


from abc import ABCMeta
from simpleml.registries.registry import Registry


# Importable registry for all custom keras objects
# Keras has an annoying persistence pattern that only supports native class references
# Custom class objects need to be passed in at load time
KERAS_REGISTRY = Registry()


class KerasRegistry(ABCMeta):
    def __new__(cls, clsname, bases, attrs):
        newclass = super(KerasRegistry, cls).__new__(cls, clsname, bases, attrs)
        KERAS_REGISTRY.register(newclass)
        return newclass
