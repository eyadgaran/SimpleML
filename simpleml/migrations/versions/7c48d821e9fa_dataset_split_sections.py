"""dataset split sections

Revision ID: 7c48d821e9fa
Revises: k65erd8bf5d0
Create Date: 2021-08-31 19:48:21.732924

Data only migration
"""
import logging
import os
from alembic import op
from sqlalchemy import MetaData, Column, ForeignKey, String, Boolean
from sqlalchemy.orm import scoped_session, sessionmaker, relationship

from simpleml.registries import SIMPLEML_REGISTRY
from simpleml.persistables.sqlalchemy_types import GUID, MutableJSON
from simpleml.persistables.base_sqlalchemy import BaseSQLAlchemy
from simpleml.persistables.hashing import CustomHasherMixin
from simpleml.pipelines import ExplicitSplitPipeline, RandomSplitPipeline
from simpleml.metrics.classification import ClassificationMetric


LOGGER = logging.getLogger(__name__)

# revision identifiers, used by Alembic.
revision = '7c48d821e9fa'
down_revision = 'k65erd8bf5d0'
branch_labels = None
depends_on = None


class UpgradeTableModel(BaseSQLAlchemy):
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
    has_external_files = Column(Boolean, default=False)
    filepaths = Column(MutableJSON, default={})


class UpgradeDatasetModel(UpgradeTableModel):
    __tablename__ = 'datasets'


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
    has_external_files = Column(Boolean, default=False)
    filepaths = Column(MutableJSON, default={})


class DowngradeDatasetModel(DowngradeTableModel):
    __tablename__ = 'datasets'


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
    LOGGER.info("Running data only migration 7c48d821e9fa")
    connection = op.get_bind()
    session = configure_session(connection, upgrade_op=True)
    try:
        upgrade_data(session, UpgradeDatasetModel)
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

    # allow user to skip hash modification
    skip_hash_update = 'SIMPLEML_SKIP_HASH_MIGRATION_7c48d821e9fa' in os.environ
    if skip_hash_update:
        LOGGER.info("User env variable set. Skipping hash changes for revision 7c48d821e9fa")
    else:
        LOGGER.info("User env variable not set (`SIMPLEML_SKIP_HASH_MIGRATION_7c48d821e9fa`). Running hash changes for revision 7c48d821e9fa")

    for record in records:
        label_columns = record.metadata_['config'].pop('label_columns', [])
        record.metadata_['config']['split_section_map'] = {'y': label_columns}

        if not skip_hash_update:
            record.hash_ = new_hash(record)

    if records:
        session.add_all(records)


def downgrade():
    LOGGER.info("Running data only migration 7c48d821e9fa")
    connection = op.get_bind()
    session = configure_session(connection, upgrade_op=False)
    try:
        downgrade_data(session, DowngradeDatasetModel)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def downgrade_data(session, table):
    # get all nonnull records
    records = table.all()
    LOGGER.info(f'Modifying data for {len(records)} records - can be a lossy change!')

    # allow user to skip hash modification
    skip_hash_update = 'SIMPLEML_SKIP_HASH_MIGRATION_7c48d821e9fa' in os.environ
    if skip_hash_update:
        LOGGER.info("User env variable set. Skipping hash changes for revision 7c48d821e9fa")
    else:
        LOGGER.info("User env variable not set (`SIMPLEML_SKIP_HASH_MIGRATION_7c48d821e9fa`). Running hash changes for revision 7c48d821e9fa")

    for record in records:
        split_sections = record.metadata_['config'].pop('split_section_map', {})
        record.metadata_['config']['label_columns'] = split_sections.get('y', [])

        if not skip_hash_update:
            record.hash_ = new_hash(record)

    if records:
        session.add_all(records)


def new_hash(record):
    try:
        # turn record into a persistable with a hash method
        record_cls = SIMPLEML_REGISTRY.get(record.registered_name)
        record.__class__ = record_cls
        if not isinstance(record.filepaths, dict):
            record.unloaded_artifacts = []
        else:
            record.unloaded_artifacts = list(self.filepaths.keys())

        return record._hash()
    except Exception as e:
        LOGGER.error(f"Failed to generate a new hash for record, skipping modification; {e}")
        return record.hash_
