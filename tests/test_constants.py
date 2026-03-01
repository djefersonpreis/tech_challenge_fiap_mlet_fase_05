"""Tests for utility constants module."""

import pytest
from pathlib import Path

from src.utils.constants import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    DESTAQUE_MAPPING,
    GENDER_MAPPING,
    INSTITUTION_MAPPING,
    NUMERIC_FEATURES,
    PEDRA_ORDER,
    PROJECT_ROOT,
    REC_PSICOLOGIA_MAPPING,
    TARGET_COL,
    YES_NO_MAPPING,
)


class TestConstants:
    """Tests for project constants."""

    def test_project_root_exists(self):
        assert PROJECT_ROOT.exists()

    def test_all_features_is_union(self):
        assert ALL_FEATURES == NUMERIC_FEATURES + CATEGORICAL_FEATURES

    def test_target_col_defined(self):
        assert TARGET_COL == "Defas_bin"

    def test_pedra_order_has_4_levels(self):
        assert len(PEDRA_ORDER) == 4
        assert PEDRA_ORDER["Quartzo"] < PEDRA_ORDER["Topázio"]

    def test_destaque_mapping(self):
        assert DESTAQUE_MAPPING["Destaque"] == 1
        assert DESTAQUE_MAPPING["Melhorar"] == 0

    def test_gender_mapping(self):
        assert len(GENDER_MAPPING) == 2

    def test_institution_mapping(self):
        assert len(INSTITUTION_MAPPING) == 3

    def test_yes_no_mapping(self):
        assert YES_NO_MAPPING["Sim"] == 1
        assert YES_NO_MAPPING["Não"] == 0

    def test_rec_psicologia_mapping(self):
        assert len(REC_PSICOLOGIA_MAPPING) == 5

    def test_numeric_features_count(self):
        assert len(NUMERIC_FEATURES) == 9

    def test_categorical_features_count(self):
        assert len(CATEGORICAL_FEATURES) == 9
