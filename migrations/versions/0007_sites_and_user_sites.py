"""site-level camera permissions: cameras.site, users.sites

Revision ID: 0007
Revises: 0006
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '0007'
down_revision = '0006'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('cameras', sa.Column('site', sa.String(length=64), nullable=False, server_default='default'))
    op.create_index('ix_cameras_site', 'cameras', ['site'])
    op.add_column('users', sa.Column('sites', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False, server_default='[]'))


def downgrade() -> None:
    op.drop_column('users', 'sites')
    op.drop_index('ix_cameras_site', table_name='cameras')
    op.drop_column('cameras', 'site')
