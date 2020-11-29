'''
Base Module for Datasets

Two use cases:
    1) Processed, or traditional datasets. In situations of clean,
representative data, this can be used directly for modeling purposes.

    2) Otherwise, a `raw dataset` needs to be created first with a `dataset pipeline`
to transform it into the processed form.
'''

__author__ = 'Elisha Yadgaran'


from simpleml.persistables.base_persistable import Persistable
from simpleml.save_patterns.decorators import ExternalArtifactDecorators
from simpleml.persistables.sqlalchemy_types import GUID
from simpleml.registries import DatasetRegistry

from future.utils import with_metaclass
from sqlalchemy import Column, ForeignKey, UniqueConstraint, Index
from sqlalchemy.orm import relationship
import logging


LOGGER = logging.getLogger(__name__)


@ExternalArtifactDecorators.register_artifact(
    artifact_name='dataset', save_attribute='dataframe', restore_attribute='_external_file')
class AbstractDataset(with_metaclass(DatasetRegistry, Persistable)):
    '''
    Abstract Base class for all Dataset objects.

    Every dataset has a "dataframe" object associated with it that is responsible
    for housing the data. The term dataframe is a bit of a misnomer since it
    does not need to be a pandas.DataFrame obejct.

    Each dataframe can be subdivided by inheriting classes and mixins to support
    numerous representations:
    ex: y column for supervised
        train/test/validation splits
        ...

    Datasets can be constructed from scratch or as derivatives of existing datasets.
    In the event of derivation, a pipeline must be specified to transform the
    original data

    -------
    Schema
    -------
    No additional columns
    '''

    __abstract__ = True

    object_type = 'DATASET'

    def __init__(self, has_external_files=True, label_columns=None, **kwargs):
        # If no save patterns are set, specify a default for disk_pickled
        if 'save_patterns' not in kwargs:
            kwargs['save_patterns'] = {'dataset': ['disk_pickled']}
        super(AbstractDataset, self).__init__(
            has_external_files=has_external_files, **kwargs)

        # By default assume unsupervised so no targets
        if label_columns is None:
            label_columns = []
        self.config['label_columns'] = label_columns

    @property
    def dataframe(self):
        # Return dataframe if generated, otherwise generate first
        self.load_if_unloaded('dataset')

        if not hasattr(self, '_external_file') or self._external_file is None:
            self.build_dataframe()

        return self._external_file

    @property
    def label_columns(self):
        '''
        Keep column list for labels in metadata to persist through saving
        '''
        return self.config.get('label_columns', [])

    def build_dataframe(self):
        '''
        Must set self._external_file
        Cant set as abstractmethod because of database lookup dependency
        '''
        raise NotImplementedError

    def add_pipeline(self, pipeline):
        '''
        Setter method for dataset pipeline used
        '''
        self.pipeline = pipeline

    def _hash(self):
        '''
        Datasets rely on external data so instead of hashing only the config,
        hash the actual resulting dataframe
        This requires loading the data before determining duplication
        so overwrite for differing behavior

        Technically there is little reason to hash anything besides the dataframe,
        but a design choice was made to separate the representation of the data
        from the use cases. For example: two dataset objects with different configured
        labels will yield different downstream results, even if the data is identical.
        Similarly with the pipeline, maintaining the back reference to the originating
        source is preferred, even if the final data can be made through different
        transformations.

        Hash is the combination of the:
            1) Dataframe
            2) Config
            3) Pipeline
        '''
        dataframe = self.dataframe
        config = self.config
        if self.pipeline is not None:
            pipeline_hash = self.pipeline.hash_ or self.pipeline._hash()
        else:
            pipeline_hash = None

        return self.custom_hasher((dataframe, config, pipeline_hash))

    def save(self, **kwargs):
        '''
        Extend parent function with a few additional save routines
        '''
        if self.pipeline is None:
            LOGGER.warning('Not using a dataset pipeline. Correct if this is unintended')

        super(AbstractDataset, self).save(**kwargs)

        # Sqlalchemy updates relationship references after save so reload class
        if self.pipeline:
            self.pipeline.load(load_externals=False)

    def load(self, **kwargs):
        '''
        Extend main load routine to load relationship class
        '''
        super(AbstractDataset, self).load(**kwargs)

        # By default dont load data unless it actually gets used
        if self.pipeline:
            self.pipeline.load(load_externals=False)


class Dataset(AbstractDataset):
    '''
    Base class for all  Dataset objects.

    -------
    Schema
    -------
    pipeline_id: foreign key relation to the dataset pipeline used as input
    '''
    __tablename__ = 'datasets'

    pipeline_id = Column(GUID, ForeignKey("pipelines.id"))
    pipeline = relationship("Pipeline", enable_typechecks=False, foreign_keys=[pipeline_id])

    __table_args__ = (
        # Unique constraint for versioning
        UniqueConstraint('name', 'version', name='dataset_name_version_unique'),
        # Index for searching through friendly names
        Index('dataset_name_index', 'name'),
    )
