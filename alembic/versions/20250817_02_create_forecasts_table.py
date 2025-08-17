"""create forecasts table if missing

Revision ID: 20250817_02
Revises: 20250817_01
Create Date: 2025-08-17
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '20250817_02'
down_revision = '20250817_01'
branch_labels = None
depends_on = None

TABLE_NAME = 'forecasts'
INDEX_NAME = 'idx_forecasts_equipment_created'


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    tables = insp.get_table_names()
    if TABLE_NAME not in tables:
        op.create_table(
            TABLE_NAME,
            sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.Column('raw_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('raw_signals.id', ondelete='SET NULL'), nullable=True),
            sa.Column('equipment_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('equipment.id', ondelete='CASCADE'), nullable=True),
            sa.Column('horizon', sa.Integer(), nullable=False, server_default='24'),
            sa.Column('method', sa.String(length=50), nullable=False, server_default='simple_trend'),
            sa.Column('forecast_data', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
            sa.Column('probability_over_threshold', sa.Numeric(precision=5, scale=4), nullable=True),
            sa.Column('model_version', sa.String(length=20), nullable=True),
            sa.Column('risk_score', sa.Numeric(precision=5, scale=4), nullable=True),
        )
        op.create_index(INDEX_NAME, TABLE_NAME, ['equipment_id', 'created_at'])
    else:
        # Убедимся, что индекс существует
        existing_indexes = [ix['name'] for ix in insp.get_indexes(TABLE_NAME)]
        if INDEX_NAME not in existing_indexes:
            op.create_index(INDEX_NAME, TABLE_NAME, ['equipment_id', 'created_at'])


def downgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    tables = insp.get_table_names()
    if TABLE_NAME in tables:
        op.drop_table(TABLE_NAME)
