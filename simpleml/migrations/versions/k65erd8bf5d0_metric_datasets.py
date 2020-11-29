"""metric datasets

Revision ID: k65erd8bf5d0
Revises: gjevp90284z9
Create Date: 2020-11-04 20:31:02.849204

"""
import logging
from alembic import op
from sqlalchemy import MetaData, Column, ForeignKey, String
from sqlalchemy.orm import scoped_session, sessionmaker, relationship

from simpleml.registries import SIMPLEML_REGISTRY
from simpleml.persistables.sqlalchemy_types import GUID, MutableJSON
from simpleml.persistables.base_sqlalchemy import BaseSQLAlchemy
from simpleml.persistables.hashing import CustomHasherMixin
from simpleml.pipelines import ExplicitSplitPipeline, RandomSplitPipeline
from simpleml.metrics.classification import ClassificationMetric


LOGGER = logging.getLogger(__name__)

# revision identifiers, used by Alembic.
revision = 'k65erd8bf5d0'
down_revision = 'gjevp90284z9'
branch_labels = None
depends_on = None

'''
THIS MIGRATION IS NOT REVERSIBLE! DATA CAN BE DROPPED
'''


class UpgradeTableModel(BaseSQLAlchemy):
    '''
    Minimal table model to conduct migrations
    '''
    __abstract__ = True
    __table_args__ = {'extend_existing': True}
    metadata = MetaData()
    id = Column(GUID, primary_key=True)
    registered_name = Column(String, nullable=False)
    name = Column(String, default='default', nullable=False)
    metadata_ = Column('metadata', MutableJSON, default={})
    hash_ = Column('hash', String, nullable=False)


class DatasetModel(UpgradeTableModel):
    __tablename__ = 'datasets'
    pipeline_id = Column(GUID, ForeignKey("pipelines.id", name="datasets_pipeline_id_fkey"))
    pipeline = relationship(lambda: PipelineModel, enable_typechecks=False, foreign_keys=[pipeline_id])


class PipelineModel(UpgradeTableModel):
    __tablename__ = 'pipelines'
    dataset_id = Column(GUID, ForeignKey("datasets.id", name="pipelines_dataset_id_fkey"))
    dataset = relationship(lambda: DatasetModel, enable_typechecks=False, foreign_keys=[dataset_id])


class ModelsModel(UpgradeTableModel):
    __tablename__ = 'models'
    pipeline_id = Column(GUID, ForeignKey("pipelines.id", name="models_pipeline_id_fkey"))
    pipeline = relationship(lambda: PipelineModel, enable_typechecks=False)


class MetricsModel(UpgradeTableModel):
    __tablename__ = 'metrics'
    model_id = Column(GUID, ForeignKey("models.id", name="metrics_model_id_fkey"))
    model = relationship(lambda: ModelsModel, enable_typechecks=False)
    dataset_id = Column(GUID, ForeignKey("datasets.id", name="metrics_dataset_id_fkey"))
    dataset = relationship(lambda: DatasetModel, enable_typechecks=False)


class DowngradeTableModel(BaseSQLAlchemy):
    '''
    Minimal table model to conduct migrations
    '''
    __abstract__ = True
    __table_args__ = {'extend_existing': True}
    metadata = MetaData()
    id = Column(GUID, primary_key=True)
    registered_name = Column(String, nullable=False)
    metadata_ = Column('metadata', MutableJSON, default={})
    hash_ = Column('hash', String, nullable=False)


class DowngradeModelsModel(DowngradeTableModel):
    __tablename__ = 'models'


class DowngradeMetricsModel(DowngradeTableModel):
    __tablename__ = 'metrics'
    model_id = Column(GUID, ForeignKey("models.id", name="metrics_model_id_fkey"))
    model = relationship(lambda: DowngradeModelsModel, enable_typechecks=False)


def configure_session(connection, upgrade_op: bool):
    if upgrade_op:
        model = UpgradeTableModel
    else:
        model = DowngradeTableModel
    session = scoped_session(sessionmaker(autocommit=False,
                                          autoflush=False,
                                          bind=connection))
    model.metadata.bind = connection
    model.query = session.query_property()
    model.set_session(session)
    return session


def upgrade():
    connection = op.get_bind()
    session = configure_session(connection, upgrade_op=True)
    try:
        op.add_column('metrics', Column('dataset_id', GUID(), nullable=True))
        with op.batch_alter_table('metrics') as batch_op:
            batch_op.create_foreign_key('metrics_dataset_id_fkey', 'datasets',
                                        ['dataset_id'], ['id'], ondelete='CASCADE')
        upgrade_data(session, MetricsModel)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def upgrade_data(session, table):
    # get all nonnull records
    records = table.all()
    LOGGER.info(f'Modifying data for {len(records)} records')
    for record in records:
        # Retrieve original dataset id. If tweaked to not use the related
        # dataset (eg swap out ref during compute and expire before saving)
        # make sure to manually update refs
        record.dataset = record.model.pipeline.dataset
        upgrade_hash(record)

    if records:
        session.add_all(records)


def upgrade_hash(record):
    '''
    Hash computation changes because of reference to the dataset
    '''
    record_cls = SIMPLEML_REGISTRY.get(record.registered_name)
    # Only affects classification metrics
    if not issubclass(record_cls, ClassificationMetric):
        return

    # Infer the dataset split
    # split must be one of TRAIN, VALIDATION, TEST
    if record.name.startswith('in_sample_'):
        dataset_split = 'TRAIN'
    elif record.name.startswith('validation_'):
        dataset_split = 'VALIDATION'
    elif issubclass(SIMPLEML_REGISTRY.get(record.model.pipeline.registered_name), (ExplicitSplitPipeline, RandomSplitPipeline)):
        dataset_split = 'TEST'
    else:
        dataset_split = None
    record.metadata_['config']['dataset_split'] = dataset_split
    record.hash_ = new_hash(record)


def new_hash(self):
    '''
    Hash is the combination of the:
        1) Model
        2) Dataset
        2) Metric
        3) Config
    '''
    model_hash = self.model.hash_ or self.model._hash()
    dataset_hash = self.dataset.hash_ or self.dataset._hash()
    metric = self.__class__.__name__
    config = self.metadata_['config']

    return CustomHasherMixin().custom_hasher((model_hash, dataset_hash, metric, config))


def downgrade():
    LOGGER.warn('Running lossy downgrade. Data may be lost!')
    connection = op.get_bind()
    session = configure_session(connection, upgrade_op=False)
    try:
        with op.batch_alter_table('metrics') as batch_op:
            batch_op.drop_constraint("metrics_dataset_id_fkey", type_="foreignkey")
            batch_op.drop_column('dataset_id')
        downgrade_data(session, DowngradeMetricsModel)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def downgrade_data(session, table):
    # get all nonnull records
    records = table.all()
    LOGGER.info(f'Modifying data for {len(records)} records')
    for record in records:
        record_cls = SIMPLEML_REGISTRY.get(record.registered_name)
        # Only affects classification metrics
        if not issubclass(record_cls, ClassificationMetric):
            continue

        record.metadata_['config'].pop('dataset_split', None)
        record.hash_ = old_hash(record)

    if records:
        session.add_all(records)


def old_hash(self):
    '''
    Hash is the combination of the:
        1) Model
        2) Metric
        3) Config
    '''
    model_hash = self.model.hash_ or self.model._hash()
    metric = self.__class__.__name__
    config = self.metadata_['config']

    return CustomHasherMixin().custom_hasher((model_hash, metric, config))
