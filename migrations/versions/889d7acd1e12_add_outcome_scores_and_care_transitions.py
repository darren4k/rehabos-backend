"""add_outcome_scores_and_care_transitions

Revision ID: 889d7acd1e12
Revises:
Create Date: 2026-03-17 10:13:00.718896

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


# revision identifiers, used by Alembic.
revision: str = '889d7acd1e12'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create outcome_scores and care_transitions tables."""
    op.create_table('outcome_scores',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('patient_id', UUID(as_uuid=True), nullable=False),
    sa.Column('episode_id', sa.String(length=50), nullable=True),
    sa.Column('encounter_id', sa.String(length=50), nullable=True),
    sa.Column('measure_name', sa.String(length=30), nullable=False),
    sa.Column('score', sa.Float(), nullable=False),
    sa.Column('recorded_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('recorded_by', sa.String(length=50), nullable=True),
    sa.ForeignKeyConstraint(['patient_id'], ['patients.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_outcome_scores_patient_id', 'outcome_scores', ['patient_id'], unique=False)
    op.create_index('ix_outcome_scores_measure', 'outcome_scores', ['patient_id', 'measure_name'], unique=False)
    op.create_index('ix_outcome_scores_recorded_at', 'outcome_scores', ['recorded_at'], unique=False)

    op.create_table('care_transitions',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('transition_id', sa.String(length=50), nullable=False),
    sa.Column('patient_id', UUID(as_uuid=True), nullable=False),
    sa.Column('from_setting', sa.String(length=20), nullable=False),
    sa.Column('to_setting', sa.String(length=20), nullable=False),
    sa.Column('transition_date', sa.DateTime(timezone=True), nullable=False),
    sa.Column('reason', sa.String(length=50), nullable=False),
    sa.Column('status', sa.String(length=20), nullable=False),
    sa.Column('clinical_summary', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['patient_id'], ['patients.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('transition_id')
    )
    op.create_index('ix_care_transitions_patient_id', 'care_transitions', ['patient_id'], unique=False)
    op.create_index('ix_care_transitions_status', 'care_transitions', ['status'], unique=False)


def downgrade() -> None:
    """Drop outcome_scores and care_transitions tables."""
    op.drop_index('ix_care_transitions_status', table_name='care_transitions')
    op.drop_index('ix_care_transitions_patient_id', table_name='care_transitions')
    op.drop_table('care_transitions')
    op.drop_index('ix_outcome_scores_recorded_at', table_name='outcome_scores')
    op.drop_index('ix_outcome_scores_measure', table_name='outcome_scores')
    op.drop_index('ix_outcome_scores_patient_id', table_name='outcome_scores')
    op.drop_table('outcome_scores')
