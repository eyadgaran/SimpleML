'''
Module with helper classes to create new persistables
'''
from abc import ABCMeta, abstractmethod
from simpleml.persistables.meta_registry import SIMPLEML_REGISTRY
from simpleml.datasets.raw_datasets.base_raw_dataset import BaseRawDataset
from simpleml.pipelines.dataset_pipelines.base_dataset_pipeline import BaseDatasetPipeline
from simpleml.datasets.processed_datasets.base_processed_dataset import BaseProcessedDataset
from simpleml.pipelines.production_pipelines.base_production_pipeline import BaseProductionPipeline
import logging

LOGGER = logging.getLogger(__name__)

__author__ = 'Elisha Yadgaran'


class PersistableCreator(object):
    __metaclass__ = ABCMeta

    @classmethod
    def retrieve_or_create(self, **kwargs):
        '''
        Wrapper method to first attempt to retrieve a matching persistable and
        then create a new one if it isn't found
        '''
        cls, filters = self.determine_filters(**kwargs)
        persistable = self.retrieve(cls, filters)

        if persistable is not None:
            LOGGER.info('Using existing persistable: {}, {}, {}'.format(cls.__tablename__, persistable.name, persistable.version))
            persistable.load()
            return persistable

        else:
            LOGGER.info('Existing persistable not found. Creating new one now')
            persistable = self.create_new(**kwargs)
            LOGGER.info('Using new persistable: {}, {}, {}'.format(cls.__tablename__, persistable.name, persistable.version))
            return persistable

    @staticmethod
    def retrieve(cls, filters):
        '''
        Query database using the table model (cls) and filters for a matching
        persistable
        '''
        return cls.where(**filters).order_by(cls.version.desc()).first()

    @abstractmethod
    def determine_filters(**kwargs):
        '''
        stateless method to determine which filters to apply when looking for
        existing persistable

        Default design iterates through 2 (or 3) options when retrieving persistables:
            1) By name and version (unique properties that define persistables)
            2) By name, registered_name, and computed hash
            2.5) Optionally, just use name and registered_name (assumes class
                definition is the same and would result in an identical persistable)

        Returns: database class, filter dictionary
        '''

    @abstractmethod
    def create_new(**kwargs):
        '''
        Stateless method to create a new persistable with the desired parameters
        kwargs are passed directly to persistable
        '''


class RawDatasetCreator(PersistableCreator):
    @staticmethod
    def determine_filters(name='', version=None, strict=True, **kwargs):
        '''
        stateless method to determine which filters to apply when looking for
        existing persistable

        Returns: database class, filter dictionary

        :param registered_name: Class name registered in SimpleML
        :param strict: Specific to datasets, whether to assume same class and
        name = same persistable, or, load the data and compare the hash
        '''
        if version is not None:
            filters = {
                'name': name,
                'version': version
            }
        # Datasets are special because we cannot assert the data is the same until we load it
        elif strict:
            registered_name = kwargs.get('registered_name')
            new_dataset = SIMPLEML_REGISTRY.get(registered_name)(name=name, **kwargs)
            filters = {
                'name': name,
                'registered_name': registered_name,
                'hash_': new_dataset._hash()
            }

        else:
            filters =  {
                'name': name,
                'registered_name': kwargs.get('registered_name')
            }

        return BaseRawDataset, filters

    @staticmethod
    def create_new(registered_name, **kwargs):
        '''
        Stateless method to create a new persistable with the desired parameters
        kwargs are passed directly to persistable

        :param registered_name: Class name registered in SimpleML
        '''
        new_dataset = SIMPLEML_REGISTRY.get(registered_name)(**kwargs)
        new_dataset.build_dataframe()
        new_dataset.save()

        return new_dataset


class DatasetPipelineCreator(PersistableCreator):
    @classmethod
    def determine_filters(cls, name='', version=None, **kwargs):
        '''
        stateless method to determine which filters to apply when looking for
        existing persistable

        Returns: database class, filter dictionary

        :param registered_name: Class name registered in SimpleML
        :param dataset: dataset class or registered name
        '''
        if version is not None:
            filters = {
                'name': name,
                'version': version
            }

        else:
            dataset = kwargs.pop('raw_dataset', None)
            registered_name = kwargs.pop('registered_name')
            if dataset is None:
                dataset = cls.retrieve_dataset(**kwargs.pop('raw_dataset_kwargs', {}))

            new_pipeline = SIMPLEML_REGISTRY.get(registered_name)(name=name, **kwargs)
            new_pipeline.add_dataset(dataset)
            new_pipeline.fit()

            filters = {
                'name': name,
                'registered_name': registered_name,
                'hash_': new_pipeline._hash()
            }

        return BaseDatasetPipeline, filters

    @classmethod
    def create_new(cls, registered_name, raw_dataset=None, **kwargs):
        '''
        Stateless method to create a new persistable with the desired parameters
        kwargs are passed directly to persistable

        :param registered_name: Class name registered in SimpleML
        :param dataset: dataset class or registered name
        '''
        if raw_dataset is None:
            raw_dataset = cls.retrieve_dataset(**kwargs.pop('raw_dataset_kwargs', {}))

        new_pipeline = SIMPLEML_REGISTRY.get(registered_name)(**kwargs)
        new_pipeline.add_dataset(raw_dataset)
        new_pipeline.fit()
        new_pipeline.save()

        return new_pipeline

    @staticmethod
    def retrieve_dataset(**dataset_kwargs):
        return RawDatasetCreator.retrieve(
            *RawDatasetCreator.determine_filters(**dataset_kwargs))


class DatasetCreator(PersistableCreator):
    @classmethod
    def determine_filters(cls, name='', version=None, strict=True, **kwargs):
        '''
        stateless method to determine which filters to apply when looking for
        existing persistable

        Returns: database class, filter dictionary

        :param registered_name: Class name registered in SimpleML
        :param dataset_pipeline: dataset pipeline class or registered name
        :param strict: Specific to datasets, whether to assume same class and
        name = same persistable, or, load the data and compare the hash
        '''

        if version is not None:
            filters = {
                'name': name,
                'version': version
            }

        else:
            dataset_pipeline = kwargs.pop('dataset_pipeline', None)
            registered_name = kwargs.pop('registered_name')

            if dataset_pipeline is None:
                dataset_pipeline = cls.retrieve_pipeline(**kwargs.pop('dataset_pipeline_kwargs', {}))

            if strict:
                new_dataset = SIMPLEML_REGISTRY.get(registered_name)(name=name, **kwargs)
                new_dataset.add_pipeline(dataset_pipeline)

                filters = {
                    'name': name,
                    'registered_name': registered_name,
                    'hash_': new_dataset._hash()
                }

            else:
                filters =  {
                    'name': name,
                    'registered_name': registered_name,
                    'pipeline_id': dataset_pipeline.id
                }

        return BaseProcessedDataset, filters

    @classmethod
    def create_new(cls, registered_name, dataset_pipeline=None, **kwargs):
        '''
        Stateless method to create a new persistable with the desired parameters
        kwargs are passed directly to persistable

        :param registered_name: Class name registered in SimpleML
        :param dataset_pipeline: dataset pipeline class or registered name
        '''
        if dataset_pipeline is None:
            dataset_pipeline = cls.retrieve_pipeline(**kwargs.pop('dataset_pipeline_kwargs', {}))

        new_dataset = SIMPLEML_REGISTRY.get(registered_name)(**kwargs)
        new_dataset.add_pipeline(dataset_pipeline)
        new_dataset.build_dataframe()
        new_dataset.save()

        return new_dataset

    @staticmethod
    def retrieve_pipeline(**pipeline_kwargs):
        return DatasetPipelineCreator.retrieve(
            *DatasetPipelineCreator.determine_filters(**pipeline_kwargs))


class PipelineCreator(PersistableCreator):
    @classmethod
    def determine_filters(cls, name='', version=None, **kwargs):
        '''
        stateless method to determine which filters to apply when looking for
        existing persistable

        Returns: database class, filter dictionary

        :param registered_name: Class name registered in SimpleML
        :param dataset: dataset class or registered name
        '''
        if version is not None:
            filters = {
                'name': name,
                'version': version
            }

        else:
            dataset = kwargs.pop('dataset', None)
            registered_name = kwargs.pop('registered_name')
            if dataset is None:
                dataset = cls.retrieve_dataset(**kwargs.pop('dataset_kwargs', {}))

            new_pipeline = SIMPLEML_REGISTRY.get(registered_name)(name=name, **kwargs)
            new_pipeline.add_dataset(dataset)
            new_pipeline.fit()

            filters = {
                'name': name,
                'registered_name': registered_name,
                'hash_': new_pipeline._hash()
            }

        return BaseProductionPipeline, filters

    @classmethod
    def create_new(cls, registered_name, dataset=None, **kwargs):
        '''
        Stateless method to create a new persistable with the desired parameters
        kwargs are passed directly to persistable

        :param registered_name: Class name registered in SimpleML
        :param dataset: dataset class or registered name
        '''
        if dataset is None:
            dataset = cls.retrieve_dataset(**kwargs.pop('dataset_kwargs', {}))

        new_pipeline = SIMPLEML_REGISTRY.get(registered_name)(**kwargs)
        new_pipeline.add_dataset(dataset)
        new_pipeline.fit()
        new_pipeline.save()

        return new_pipeline

    @staticmethod
    def retrieve_dataset(**dataset_kwargs):
        return DatasetCreator.retrieve(
            *DatasetCreator.determine_filters(**dataset_kwargs))


class ModelCreator(PersistableCreator):
    pass


class MetricCreator(PersistableCreator):
    pass
