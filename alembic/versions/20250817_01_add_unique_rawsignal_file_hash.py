"""add unique index on raw_signals.file_hash

Revision ID: 20250817_01
Revises: 
Create Date: 2025-08-17
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '20250817_01'
down_revision = None
branch_labels = None
depends_on = None

INDEX_NAME = 'uq_raw_signals_file_hash'
TABLE_NAME = 'raw_signals'
COLUMN_NAME = 'file_hash'


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    # Проверяем, существует ли уже индекс (для идемпотентности)
    existing = [ix['name'] for ix in insp.get_indexes(TABLE_NAME)]
    if INDEX_NAME not in existing:
        op.create_index(INDEX_NAME, TABLE_NAME, [COLUMN_NAME], unique=True)


def downgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    existing = [ix['name'] for ix in insp.get_indexes(TABLE_NAME)]
    if INDEX_NAME in existing:
        op.drop_index(INDEX_NAME, table_name=TABLE_NAME)
