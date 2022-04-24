"""pandas datasets squeezing

Revision ID: 5c2f66bdf2ae
Revises: k65erd8bf5d0
Create Date: 2021-10-31 19:58:46.361842

Data only migration
"""
import logging

from alembic import op
from sqlalchemy import Column, MetaData, String
from sqlalchemy.orm import scoped_session, sessionmaker

from simpleml.orm.base_sqlalchemy import BaseSQLAlchemy
from simpleml.orm.sqlalchemy_types import GUID, MutableJSON

LOGGER = logging.getLogger(__name__)

# revision identifiers, used by Alembic.
revision = "5c2f66bdf2ae"
down_revision = "7c48d821e9fa"
branch_labels = None
depends_on = None


class MigrationTableModel(BaseSQLAlchemy):
    """
    Minimal table model to conduct migrations
    Data only migration (no schema changes) so single model for upgrade/downgrade
    """

    __abstract__ = True
    __table_args__ = {"extend_existing": True}
    metadata = MetaData()
    id = Column(GUID, primary_key=True)
    registered_name = Column(String, nullable=False)
    metadata_ = Column("metadata", MutableJSON, default={})


class MigrationDatasetModel(MigrationTableModel):
    __tablename__ = "datasets"


def configure_session(connection):
    model = MigrationTableModel
    session = scoped_session(
        sessionmaker(autocommit=False, autoflush=False, bind=connection)
    )
    model.metadata.bind = connection
    model.query = session.query_property()
    model.set_session(session)
    return session


def upgrade():
    LOGGER.info("Running data only migration 5c2f66bdf2ae")
    LOGGER.info(
        "Will NOT recalculate hashes for existing persistables! Use `simpleml.utils.hash_recalculation.recalculate_dataset_hashes()` to recalculate"
    )
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
    LOGGER.info(f"Modifying data for {len(records)} records")

    # upgrade logic
    # Single label pandas datasets
    # doesnt already have a squeeze_return parameter
    for record in records:
        if record.registered_name == "SingleLabelPandasDataset":
            record.metadata_["config"]["squeeze_return"] = True

    if records:
        session.add_all(records)


def downgrade():
    LOGGER.info("Running data only migration 5c2f66bdf2ae")
    LOGGER.info(
        "Will NOT recalculate hashes for existing persistables! Use `simpleml.utils.hash_recalculation.recalculate_dataset_hashes()` to recalculate"
    )

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
    LOGGER.info(f"Modifying data for {len(records)} records - can be a lossy change!")

    for record in records:
        split_sections = record.metadata_["config"].pop("split_section_map", {})
        record.metadata_["config"]["label_columns"] = split_sections.get("y", [])

    if records:
        session.add_all(records)
