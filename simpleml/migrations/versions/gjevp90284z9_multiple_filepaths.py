"""multiple filepaths

Revision ID: gjevp90284z9
Revises: deefa69553d8
Create Date: 2020-05-31 18:37:02.849204

"""
import logging
from alembic import op
from sqlalchemy import MetaData, Column
from sqlalchemy.orm import scoped_session, sessionmaker

from simpleml.persistables.sqlalchemy_types import GUID, MutableJSON
from simpleml.persistables.base_sqlalchemy import BaseSQLAlchemy

LOGGER = logging.getLogger(__name__)

# revision identifiers, used by Alembic.
revision = 'gjevp90284z9'
down_revision = 'deefa69553d8'
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
    metadata_ = Column('metadata', MutableJSON, default={})
    filepaths = Column(MutableJSON)


class DatasetModel(UpgradeTableModel):
    __tablename__ = 'datasets'


class PipelineModel(UpgradeTableModel):
    __tablename__ = 'pipelines'


class ModelsModel(UpgradeTableModel):
    __tablename__ = 'models'


class MetricsModel(UpgradeTableModel):
    __tablename__ = 'metrics'


def configure_session(connection):
    session = scoped_session(sessionmaker(autocommit=False,
                                          autoflush=False,
                                          bind=connection))
    UpgradeTableModel.metadata.bind = connection
    UpgradeTableModel.query = session.query_property()
    UpgradeTableModel.set_session(session)
    return session


def upgrade():
    '''
    This is a data only migration (schemas are unchanged)

    filepaths:
        {save_pattern: [filename]}
    -> {artifact: {save_pattern: filename}}
    '''
    connection = op.get_bind()
    session = configure_session(connection)
    # Update filepaths data to nest under artifact -> save_pattern -> filepath_data
    try:
        for table, artifact in zip(
            (DatasetModel, PipelineModel, ModelsModel, MetricsModel),
            ('dataset', 'pipeline', 'model', 'metric')
        ):
            upgrade_data(session, table, artifact)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def upgrade_data(session, table, artifact):
    # get all nonnull records
    records = table.all()
    LOGGER.info(f'Modifying data for {len(records)} records')
    for record in records:
        # Modify save method in metadata
        if 'save_method' in record.metadata_.get('state', {}):
            upgrade_metadata(record, artifact)

        # Modify filepaths
        if record.filepaths:
            upgrade_filepaths(record, artifact)

    if records:
        session.add_all(records)


def upgrade_metadata(record, artifact):
    '''
    Upgrade metadata field for each record
    '''
    pattern = record.metadata_['state'].pop('save_method')
    record.metadata_['state']['save_patterns'] = {artifact: [pattern]}


def upgrade_filepaths(record, artifact):
    '''
    Upgrade filepaths field for each record
    '''
    # Nest under artifact
    filepath = {artifact: record.filepaths}
    # Format the filepath data according to the save pattern
    for save_pattern, filepath_data in filepath[artifact].items():
        if save_pattern == 'database_table':
            # Format changed from [(schema, table)] to {schema: schema, table: table}
            filepath[artifact][save_pattern] = {'schema': filepath_data[0][0], 'table': filepath_data[0][1]}
        elif save_pattern == 'database_pickled':
            # Format changed from [record_id] to record_id
            filepath[artifact][save_pattern] = filepath_data[0]
        elif save_pattern in ['disk_pickled', 'disk_hdf5', 'disk_keras_hdf5',
                              'onedrive_pickled', 'onedrive_hdf5', 'onedrive_keras_hdf5',
                              'cloud_pickled', 'cloud_hdf5', 'cloud_keras_hdf5']:
            # Format changed from [filename] to filename
            filepath[artifact][save_pattern] = filepath_data[0]
        # Else leave as is, not a natively supported save pattern
    record.filepaths = filepath


def downgrade():
    LOGGER.warn('Running lossy downgrade. Data may be lost!')
    connection = op.get_bind()
    session = configure_session(connection)
    # Update filepaths data to remove nesting under save_pattern -> filepath_data
    try:
        for table, artifact in zip(
            (DatasetModel, PipelineModel, ModelsModel, MetricsModel),
            ('dataset', 'pipeline', 'model', 'metric')
        ):
            downgrade_data(session, table, artifact)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def downgrade_data(session, table, artifact):
    # get all nonnull records
    records = table.all()
    LOGGER.info(f'Modifying data for {len(records)} records')
    for record in records:
        # Modify save method in metadata
        if 'save_patterns' in record.metadata_.get('state', {}):
            downgrade_metadata(record, artifact)
        # Modify filepaths
        if record.filepaths:
            downgrade_filepaths(record, artifact)
    if records:
        session.add_all(records)


def downgrade_metadata(record, artifact):
    '''
    Downgrade metadata field for each record
    '''
    save_patterns = record.metadata_['state'].pop('save_patterns')
    if len(save_patterns) > 1 or len(save_patterns[artifact]) > 1:
        # There are extra patterns or artifacts that arent backwards compatible
        LOGGER.error('Found non backwards compatible artifact data, this will be lost!')
    record.metadata_['state']['save_method'] = save_patterns[artifact][0]


def downgrade_filepaths(record, artifact):
    '''
    Downgrade filepaths field for each record
    '''
    # unnest
    save_pattern, filepath = list(record.filepaths[artifact].items())[0]
    # Format the filepath data according to the save pattern
    if save_pattern == 'database_table':
        # Format changed from [(schema, table)] to {schema: schema, table: table}
        filepath = [(filepath['schema'], filepath['table'])]
    elif save_pattern == 'database_pickled':
        # Format changed from [record_id] to record_id
        filepath = [filepath]
    elif save_pattern in ['disk_pickled', 'disk_hdf5', 'disk_keras_hdf5',
                          'onedrive_pickled', 'onedrive_hdf5', 'onedrive_keras_hdf5',
                          'cloud_pickled', 'cloud_hdf5', 'cloud_keras_hdf5']:
        # Format changed from [filename] to filename
        filepath = [filepath]
    # Else leave as is, not a natively supported save pattern
    record.filepaths = {save_pattern: filepath}
