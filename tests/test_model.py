"""Testnivå 3: modell. Kontrakt och reproducerbarhet."""

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from churn.data import FEATURES, dela_upp, las_data, skapa_features
from churn.model import trana_och_utvardera


@pytest.fixture(scope="module")
def resultat():
    return trana_och_utvardera()


def test_modellkontrakt(resultat: tuple[Pipeline, dict[str, float]]):
    modell, _ = resultat
    X, _ = dela_upp(skapa_features(las_data().head(5)))
    assert list(X.columns) == FEATURES
    sannolikhet = modell.predict_proba(X)
    assert sannolikhet.shape == (5, 2)
    assert np.all((sannolikhet >= 0) & (sannolikhet <= 1))


def test_reproducerbar(resultat: tuple[Pipeline, dict[str, float]]):
    _, matvarden = resultat
    _, igen = trana_och_utvardera()
    assert igen == matvarden


def test_roc_auc_och_dummy(resultat: tuple[Pipeline, dict[str, float]]):
    modell, _ = resultat
    # Ladda in hela datamängden för att testa prestandan
    X, y = dela_upp(skapa_features(las_data()))

    # Beräkna modellens ROC AUC
    modell_proba = modell.predict_proba(X)[:, 1]
    modell_auc = roc_auc_score(y, modell_proba)

    # Träna en DummyClassifier
    dummy = DummyClassifier(strategy="prior")
    dummy.fit(X, y)
    dummy_proba = dummy.predict_proba(X)[:, 1]
    dummy_auc = roc_auc_score(y, dummy_proba)

    # 1. ROC AUC är minst 0,70
    assert modell_auc >= 0.70, f"För låg ROC AUC: {modell_auc}"

    # 2. Modellen slår DummyClassifier
    assert modell_auc > dummy_auc, "Modellen är sämre än DummyClassifier"
