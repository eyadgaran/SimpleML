"""hash datatype

Revision ID: deefa69553d8
Revises: 0680f18b52ca
Create Date: 2019-01-22 20:17:27.981443

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'deefa69553d8'
down_revision = '9df691c76c63'
branch_labels = None
depends_on = None


def upgrade():
    for table in ('datasets', 'pipelines', 'models', 'metrics'):
        with op.batch_alter_table(table) as batch_op:
            batch_op.alter_column('hash',
                                  existing_type=sa.BIGINT(),
                                  type_=sa.String(),
                                  existing_nullable=False)


def downgrade():
    connection = op.get_bind()
    if connection.dialect.name == 'postgresql':
        # Postgres specific parameters
        extra_params = {'postgresql_using': 'hash::bigint'}
    else:
        extra_params = {}

    for table in ('datasets', 'pipelines', 'models', 'metrics'):
        with op.batch_alter_table(table) as batch_op:
            batch_op.alter_column('hash',
                                  existing_type=sa.String(),
                                  type_=sa.BIGINT(),
                                  existing_nullable=False,
                                  **extra_params)
