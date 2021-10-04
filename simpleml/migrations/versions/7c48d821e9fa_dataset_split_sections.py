"""dataset split sections

Revision ID: 7c48d821e9fa
Revises: k65erd8bf5d0
Create Date: 2021-08-31 19:48:21.732924

Data only migration
"""
import logging
import os
from alembic import op
from sqlalchemy import MetaData, Column
from sqlalchemy.orm import scoped_session, sessionmaker

from simpleml.persistables.sqlalchemy_types import GUID, MutableJSON
from simpleml.persistables.base_sqlalchemy import BaseSQLAlchemy


LOGGER = logging.getLogger(__name__)

# revision identifiers, used by Alembic.
revision = '7c48d821e9fa'
down_revision = 'k65erd8bf5d0'
branch_labels = None
depends_on = None


class MigrationTableModel(BaseSQLAlchemy):
    '''
    Minimal table model to conduct migrations
    Data only migration (no schema changes) so single model for upgrade/downgrade
    '''
    __abstract__ = True
    __table_args__ = {'extend_existing': True}
    metadata = MetaData()
    id = Column(GUID, primary_key=True)
    metadata_ = Column('metadata', MutableJSON, default={})


class MigrationDatasetModel(MigrationTableModel):
    __tablename__ = 'datasets'


def configure_session(connection):
    model = MigrationTableModel
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
    session = configure_session(connection)

    try:
        upgrade_data(session, MigrationDatasetModel)
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
        label_columns = record.metadata_['config'].pop('label_columns', [])
        record.metadata_['config']['split_section_map'] = {'y': label_columns}

    if records:
        session.add_all(records)


def downgrade():
    LOGGER.info("Running data only migration 7c48d821e9fa")
    connection = op.get_bind()
    session = configure_session(connection)

    try:
        downgrade_data(session, MigrationDatasetModel)
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

    for record in records:
        split_sections = record.metadata_['config'].pop('split_section_map', {})
        record.metadata_['config']['label_columns'] = split_sections.get('y', [])

    if records:
        session.add_all(records)
