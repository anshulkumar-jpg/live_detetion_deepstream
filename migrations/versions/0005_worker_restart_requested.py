"""workers.restart_requested_at: the API records a restart request, the scheduler (which signs commands) carries it out

Revision ID: 0005
Revises: 0004
"""
from alembic import op
import sqlalchemy as sa

revision = '0005'
down_revision = '0004'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('workers', sa.Column('restart_requested_at', sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column('workers', 'restart_requested_at')
