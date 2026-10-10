"""Testnivå 1: kod. Enhetstester av feature-koden, utan modell och utan riktig data."""

import pandas as pd
from pandas import DataFrame
import pytest

from churn.data import skapa_features


@pytest.fixture
def liten_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "manadskostnad": [299.0, 149.0],
            "data_gb_per_manad": [9.0, 0.0],
            "region": [" stockholm", "Väst"],
        }
    )


def test_kostnad_per_gb(liten_df: DataFrame):
    ut = skapa_features(liten_df)
    assert ut["kostnad_per_gb"].tolist() == pytest.approx([29.9, 149.0])


def test_region_normaliseras(liten_df: DataFrame):
    assert skapa_features(liten_df)["region"].tolist() == ["Stockholm", "Väst"]


def test_indata_andras_inte(liten_df: DataFrame):
    original = liten_df.copy()
    skapa_features(liten_df)
    pd.testing.assert_frame_equal(liten_df, original)

def test_region_versaler_blir_sma_bokstaver():
    # Skapa data med versaler OCH de andra kolumnerna som funktionen kräver
    df = pd.DataFrame({
        "region": ["STOCKHOLM", "GÖTEBORG"],
        "manadskostnad": [299.0, 149.0],
        "data_gb_per_manad": [9.0, 0.0]
    })
    ut = skapa_features(df)

    # Uppgiften kräver att t.ex. 'STOCKHOLM' blir 'stockholm'
    assert ut["region"].tolist() == ["Stockholm", "Göteborg"]
