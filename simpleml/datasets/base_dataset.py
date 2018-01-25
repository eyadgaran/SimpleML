from simpleml.persistables.base_persistable import BasePersistable


__author__ = 'Elisha Yadgaran'


class BaseDataset(BasePersistable):
    '''
    Base class for all Dataset objects.

    -------
    Schema
    -------

    '''
    __tablename__ = 'datasets'
