"""per-camera class selection (camera_model_assignments.detect_class_ids)

Revision ID: 0003
Revises: 0002
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '0003'
down_revision = '0002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('camera_model_assignments',
                  sa.Column('detect_class_ids', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'),
                            nullable=True))


def downgrade() -> None:
    op.drop_column('camera_model_assignments', 'detect_class_ids')
