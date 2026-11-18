"""per-camera detection confidence and NMS IoU (camera_model_assignments)

Revision ID: 0004
Revises: 0003
"""
from alembic import op
import sqlalchemy as sa

revision = '0004'
down_revision = '0003'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('camera_model_assignments', sa.Column('conf_threshold', sa.Float(), nullable=True))
    op.add_column('camera_model_assignments', sa.Column('iou_threshold', sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column('camera_model_assignments', 'iou_threshold')
    op.drop_column('camera_model_assignments', 'conf_threshold')
