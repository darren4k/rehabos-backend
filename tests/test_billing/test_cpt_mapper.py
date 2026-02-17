"""Tests for CPT code mapping from interventions."""

import pytest

from rehab_os.billing.cpt_mapper import (
    CPT_CODES,
    INTERVENTION_TO_CPT,
    map_interventions_to_cpt,
)


class TestInterventionMapping:
    def test_therapeutic_exercise_maps_to_97110(self):
        result = map_interventions_to_cpt([{"name": "therapeutic exercise"}])
        assert len(result) == 1
        assert result[0].code == "97110"

    def test_gait_training_maps_to_97116(self):
        result = map_interventions_to_cpt([{"name": "gait training"}])
        assert result[0].code == "97116"

    def test_balance_training_maps_to_97530(self):
        result = map_interventions_to_cpt([{"name": "balance training"}])
        assert result[0].code == "97530"

    def test_manual_therapy_maps_to_97140(self):
        result = map_interventions_to_cpt([{"name": "manual therapy"}])
        assert result[0].code == "97140"

    def test_neuro_re_ed_maps_to_97112(self):
        result = map_interventions_to_cpt([{"name": "neuromuscular re-education"}])
        assert result[0].code == "97112"

    def test_modalities_maps_to_97010(self):
        result = map_interventions_to_cpt([{"name": "modalities"}])
        assert result[0].code == "97010"

    def test_multiple_interventions(self):
        result = map_interventions_to_cpt([
            {"name": "therapeutic exercise", "duration_minutes": 15},
            {"name": "gait training", "duration_minutes": 10},
            {"name": "manual therapy", "duration_minutes": 10},
        ])
        codes = {r.code for r in result}
        assert "97110" in codes
        assert "97116" in codes
        assert "97140" in codes

    def test_duration_passed_through(self):
        result = map_interventions_to_cpt([{"name": "therapeutic exercise", "duration_minutes": 20}])
        assert result[0].minutes == 20

    def test_default_duration_15(self):
        result = map_interventions_to_cpt([{"name": "therapeutic exercise"}])
        assert result[0].minutes == 15

    def test_duplicate_codes_merge_duration(self):
        # "strengthening" and "stretching" both map to 97110
        result = map_interventions_to_cpt([
            {"name": "strengthening", "duration_minutes": 10},
            {"name": "stretching", "duration_minutes": 10},
        ])
        lines_97110 = [r for r in result if r.code == "97110"]
        assert len(lines_97110) == 1
        assert lines_97110[0].minutes == 20  # merged

    def test_unknown_intervention_skipped(self):
        result = map_interventions_to_cpt([{"name": "hyperbaric oxygen therapy"}])
        assert len(result) == 0

    def test_eval_note_adds_eval_code(self):
        result = map_interventions_to_cpt(
            [{"name": "therapeutic exercise"}],
            note_type="initial_evaluation",
        )
        codes = {r.code for r in result}
        assert "97162" in codes  # PT eval moderate complexity

    def test_recert_note_adds_reeval_code(self):
        result = map_interventions_to_cpt(
            [{"name": "therapeutic exercise"}],
            note_type="recertification",
        )
        codes = {r.code for r in result}
        assert "97164" in codes  # PT re-evaluation


class TestCPTCodesTable:
    def test_all_common_codes_present(self):
        expected = ["97110", "97112", "97116", "97140", "97530", "97535", "97010", "97035"]
        for code in expected:
            assert code in CPT_CODES, f"Missing CPT code: {code}"

    def test_timed_vs_untimed(self):
        assert CPT_CODES["97110"].category == "timed"
        assert CPT_CODES["97010"].category == "untimed"

    def test_all_interventions_have_cpt(self):
        for intervention, codes in INTERVENTION_TO_CPT.items():
            for code in codes:
                assert code in CPT_CODES, f"Intervention '{intervention}' references unknown CPT: {code}"
