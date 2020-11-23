"""metric datasets

Revision ID: k65erd8bf5d0
Revises: gjevp90284z9
Create Date: 2020-11-04 20:31:02.849204

"""
import logging
from alembic import op
from sqlalchemy import MetaData, Column, ForeignKey
from sqlalchemy.orm import scoped_session, sessionmaker, relationship

from simpleml.persistables.sqlalchemy_types import GUID
from simpleml.persistables.base_sqlalchemy import BaseSQLAlchemy

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


def configure_session(connection):
    session = scoped_session(sessionmaker(autocommit=False,
                                          autoflush=False,
                                          bind=connection))
    UpgradeTableModel.metadata.bind = connection
    UpgradeTableModel.query = session.query_property()
    UpgradeTableModel.set_session(session)
    return session


def upgrade():
    connection = op.get_bind()
    session = configure_session(connection)
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
        record.dataset_id = record.model.pipeline.dataset.id

    if records:
        session.add_all(records)


def downgrade():
    LOGGER.warn('Running lossy downgrade. Data may be lost!')
    with op.batch_alter_table('metrics') as batch_op:
        batch_op.drop_constraint("metrics_dataset_id_fkey", type_="foreignkey")
        batch_op.drop_column('dataset_id')
