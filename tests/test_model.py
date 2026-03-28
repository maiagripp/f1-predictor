"""
Testes automatizados PyTest — F1 Race Winner Predictor
Valida que o modelo atende aos requisitos mínimos de performance antes do deploy.
"""
import pickle
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score

# ── Thresholds de performance ────────────────────────────────────────────────
# Com ~5% de classe positiva (1 vencedor por corrida), F1 e AUC são as métricas
# mais relevantes. Thresholds conservadores mas realistas para um dataset desbalanceado.
THRESHOLD_F1        = 0.40   # ≥ 40% F1-Score 
THRESHOLD_RECALL    = 0.35   # ≥ 35% Recall 
THRESHOLD_AUC_ROC   = 0.75   # ≥ 75% AUC-ROC 
THRESHOLD_PRECISION = 0.30   # ≥ 30% Precision

MODEL_PATH = Path(__file__).parent.parent / "test_requirements" / "model.pkl"
DATA_URL = "https://raw.githubusercontent.com/maiagripp/f1-predictor/refs/heads/main/f1_enhanced_dataset_for_analysis.csv"

FEATURES_NUM = ['QualifyingPosition', 'StartPosition', 'PitStopCount']
FEATURES_CAT = ['Weather', 'TireStrategy', 'Team']


@pytest.fixture(scope="module")
def model():
    assert MODEL_PATH.exists(), f"Modelo não encontrado: {MODEL_PATH}"
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def test_data():
    df = pd.read_csv(DATA_URL)
    df['winner'] = (df['FinishPosition'] == 1).astype(int)

    from sklearn.model_selection import train_test_split
    X = df[FEATURES_NUM + FEATURES_CAT]
    y = df['winner']
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_test, y_test


def test_model_loads(model):
    """Verifica que o modelo carrega corretamente e possui os métodos necessários."""
    assert model is not None
    assert hasattr(model, 'predict')
    assert hasattr(model, 'predict_proba')


def test_model_f1(model, test_data):
    """F1-Score deve ser >= THRESHOLD_F1 (métrica principal para classes desbalanceadas)."""
    X_test, y_test = test_data
    y_pred = model.predict(X_test)
    score = f1_score(y_test, y_pred, zero_division=0)
    assert score >= THRESHOLD_F1, (
        f"F1-Score {score:.4f} abaixo do threshold {THRESHOLD_F1}"
    )


def test_model_recall(model, test_data):
    """Recall deve ser >= THRESHOLD_RECALL (identificar vencedores reais)."""
    X_test, y_test = test_data
    y_pred = model.predict(X_test)
    score = recall_score(y_test, y_pred, zero_division=0)
    assert score >= THRESHOLD_RECALL, (
        f"Recall {score:.4f} abaixo do threshold {THRESHOLD_RECALL}"
    )


def test_model_auc_roc(model, test_data):
    """AUC-ROC deve ser >= THRESHOLD_AUC_ROC."""
    X_test, y_test = test_data
    y_proba = model.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, y_proba)
    assert score >= THRESHOLD_AUC_ROC, (
        f"AUC-ROC {score:.4f} abaixo do threshold {THRESHOLD_AUC_ROC}"
    )


def test_model_precision(model, test_data):
    """Precision deve ser >= THRESHOLD_PRECISION."""
    X_test, y_test = test_data
    y_pred = model.predict(X_test)
    score = precision_score(y_test, y_pred, zero_division=0)
    assert score >= THRESHOLD_PRECISION, (
        f"Precision {score:.4f} abaixo do threshold {THRESHOLD_PRECISION}"
    )


def test_output_shape(model, test_data):
    """Saída deve ter o mesmo número de amostras que a entrada."""
    X_test, _ = test_data
    assert len(model.predict(X_test)) == len(X_test)


def test_binary_classes(model, test_data):
    """Predições devem ser apenas 0 ou 1."""
    X_test, _ = test_data
    preds = model.predict(X_test)
    assert set(preds).issubset({0, 1}), "Predições fora de {0, 1}"


def test_proba_sums_to_one(model, test_data):
    """Probabilidades devem somar 1.0 para cada amostra."""
    X_test, _ = test_data
    proba = model.predict_proba(X_test)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_grid_front_wins_more(model, test_data):
    """
    Sanity check: piloto largando em P1 deve ter maior probabilidade de
    vitória do que piloto largando em P20 (mesmas condições).
    """
    base = {
        'QualifyingPosition': 1.0,
        'PitStopCount': 2.0,
        'Weather': 'Sunny',
        'TireStrategy': 'Soft-Hard',
        'Team': 'Mercedes'         
    }
    front = pd.DataFrame([{**base, 'StartPosition': 1.0}])
    back  = pd.DataFrame([{**base, 'StartPosition': 20.0}])

    prob_front = model.predict_proba(front)[0][1]
    prob_back  = model.predict_proba(back)[0][1]

    assert prob_front > prob_back, (
        f"Lewis largando em P1 deveria ter maior prob de vitória "
        f"({prob_front:.4f}) vs P20 ({prob_back:.4f})"
    )