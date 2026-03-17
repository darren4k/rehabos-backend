"""Tests for rehab_os.clinical.settings — setting-aware clinical configuration."""
from __future__ import annotations

import pytest

from rehab_os.clinical.settings import (
    SETTING_CONFIG,
    ClinicalSetting,
    get_setting_config,
    get_valid_note_types,
    requires_instrument,
)


class TestSettingEnum:
    def test_all_settings_defined(self):
        assert len(ClinicalSetting) == 7
        expected = {"outpatient", "homecare", "snf", "irf", "alf", "school", "telehealth"}
        actual = {s.value for s in ClinicalSetting}
        assert actual == expected

    def test_each_setting_has_config(self):
        for setting in ClinicalSetting:
            config = get_setting_config(setting)
            assert "note_types" in config
            assert "billing_form" in config


class TestOutpatientConfig:
    def test_no_oasis(self):
        config = get_setting_config(ClinicalSetting.OUTPATIENT)
        assert config["requires_oasis"] is False

    def test_billing_form(self):
        config = get_setting_config(ClinicalSetting.OUTPATIENT)
        assert config["billing_form"] == "CMS-1500"

    def test_scheduling_type(self):
        config = get_setting_config(ClinicalSetting.OUTPATIENT)
        assert config["scheduling_type"] == "clinic_slots"


class TestHomecareConfig:
    def test_requires_oasis(self):
        config = get_setting_config(ClinicalSetting.HOMECARE)
        assert config["requires_oasis"] is True

    def test_60_day_cert(self):
        config = get_setting_config(ClinicalSetting.HOMECARE)
        assert config["cert_period_days"] == 60
        assert config["max_episode_days"] == 60

    def test_route_based_scheduling(self):
        config = get_setting_config(ClinicalSetting.HOMECARE)
        assert config["scheduling_type"] == "route_based"


class TestSNFConfig:
    def test_requires_fim(self):
        config = get_setting_config(ClinicalSetting.SNF)
        assert config["requires_fim"] is True

    def test_billing_form(self):
        config = get_setting_config(ClinicalSetting.SNF)
        assert config["billing_form"] == "UB-04"

    def test_facility_schedule(self):
        config = get_setting_config(ClinicalSetting.SNF)
        assert config["scheduling_type"] == "facility_schedule"


class TestGetValidNoteTypes:
    def test_outpatient_note_types(self):
        types = get_valid_note_types(ClinicalSetting.OUTPATIENT)
        assert "initial_eval" in types
        assert "progress_note" in types
        assert "discharge" in types

    def test_homecare_note_types(self):
        types = get_valid_note_types(ClinicalSetting.HOMECARE)
        assert "soc_eval" in types
        assert "supervisory_visit" in types

    def test_snf_note_types(self):
        types = get_valid_note_types(ClinicalSetting.SNF)
        assert "weekly_summary" in types


class TestRequiresInstrument:
    def test_homecare_requires_oasis(self):
        assert requires_instrument(ClinicalSetting.HOMECARE, "oasis") is True

    def test_outpatient_no_oasis(self):
        assert requires_instrument(ClinicalSetting.OUTPATIENT, "oasis") is False

    def test_snf_requires_fim(self):
        assert requires_instrument(ClinicalSetting.SNF, "fim") is True

    def test_outpatient_no_fim(self):
        assert requires_instrument(ClinicalSetting.OUTPATIENT, "fim") is False

    def test_irf_requires_fim(self):
        assert requires_instrument(ClinicalSetting.IRF, "fim") is True

    def test_school_requires_iep(self):
        assert requires_instrument(ClinicalSetting.SCHOOL, "iep") is True

    def test_unknown_instrument(self):
        assert requires_instrument(ClinicalSetting.OUTPATIENT, "nonexistent") is False
