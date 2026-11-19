"""token revocation (users.token_epoch, revoked_tokens) and login throttling (login_throttle)

Revision ID: 0006
Revises: 0005
"""
from alembic import op
import sqlalchemy as sa

revision = '0006'
down_revision = '0005'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('users', sa.Column('token_epoch', sa.Integer(), nullable=False, server_default='0'))
    op.create_table(
        'revoked_tokens',
        sa.Column('jti', sa.String(length=64), primary_key=True),
        sa.Column('username', sa.String(length=64), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('revoked_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index('ix_revoked_exp', 'revoked_tokens', ['expires_at'])
    op.create_table(
        'login_throttle',
        sa.Column('key', sa.String(length=160), primary_key=True),
        sa.Column('failures', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('window_started_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('locked_until', sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_table('login_throttle')
    op.drop_index('ix_revoked_exp', table_name='revoked_tokens')
    op.drop_table('revoked_tokens')
    op.drop_column('users', 'token_epoch')
