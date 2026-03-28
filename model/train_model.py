"""
train_model.py — Treina o modelo de previsão de vencedor F1 e exporta os artefatos.
Execute: python model/train_model.py
Os arquivos model.pkl e model_metadata.pkl serão salvos em test_requirements/
"""
import os
import pickle
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import f1_score, roc_auc_score, recall_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ── Configuração de caminhos ──────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR  = os.path.join(BASE_DIR, "test_requirements")
DATA_PATH   = os.path.join(BASE_DIR, "f1_enhanced_dataset_for_analysis.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Carga dos dados ────────────────────────────────────────────────────────
print("📂 Carregando dataset...")
df = pd.read_csv(DATA_PATH)
df["winner"] = (df["FinishPosition"] == 1).astype(int)

FEATURES_NUM = ["QualifyingPosition", "StartPosition", "PitStopCount"]
FEATURES_CAT = ["Weather", "TireStrategy", "Team"]

X = df[FEATURES_NUM + FEATURES_CAT]
y = df["winner"]

print(f"   Total: {len(df)} registros | Vencedores: {y.sum()} ({y.mean():.2%})")

# ── 2. Separação treino/teste ─────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 3. Preprocessor ───────────────────────────────────────────────────────────
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), FEATURES_NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), FEATURES_CAT)
])

# ── 4. Definição dos pipelines ────────────────────────────────────────────────
pipelines = {
    "KNN": ImbPipeline([
        ("pre", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("model", KNeighborsClassifier())
    ]),
    "Decision Tree": ImbPipeline([
        ("pre", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("model", DecisionTreeClassifier(random_state=42, class_weight="balanced"))
    ]),
    "Naive Bayes": ImbPipeline([
        ("pre", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("model", GaussianNB())
    ]),
    "SVM": ImbPipeline([
        ("pre", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("model", SVC(probability=True, random_state=42, class_weight="balanced"))
    ])
}

# ── 5. GridSearchCV ───────────────────────────────────────────────────────────
param_grids = {
    "KNN":           {"model__n_neighbors": [3, 5, 7], "model__weights": ["uniform", "distance"]},
    "Decision Tree": {"model__max_depth": [3, 5, 7, None], "model__criterion": ["gini", "entropy"]},
    "Naive Bayes":   {"model__var_smoothing": [1e-9, 1e-8, 1e-7]},
    "SVM":           {"model__C": [0.1, 1, 10], "model__kernel": ["rbf", "linear"]}
}

print("\n🔧 Otimizando hiperparâmetros...")
best_models = {}
for nome, pipeline in pipelines.items():
    gs = GridSearchCV(pipeline, param_grids[nome], cv=5,
                      scoring="f1", n_jobs=-1, error_score=0)
    gs.fit(X_train, y_train)
    best_models[nome] = gs.best_estimator_
    print(f"   ✓ {nome} — F1 CV: {gs.best_score_:.4f}")

# ── 6. Avaliação no teste ─────────────────────────────────────────────────────
print("\n📊 Resultados no conjunto de teste:")
resultados = {}
for nome, modelo in best_models.items():
    y_pred  = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]
    resultados[nome] = {
        "F1":     f1_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "AUC":    roc_auc_score(y_test, y_proba)
    }
    print(f"   {nome:<18} F1={resultados[nome]['F1']:.4f}  "
          f"Recall={resultados[nome]['Recall']:.4f}  "
          f"AUC={resultados[nome]['AUC']:.4f}")

# ── 7. Seleção do melhor modelo ───────────────────────────────────────────────
melhor_nome  = max(resultados, key=lambda k: resultados[k]["F1"])
melhor_modelo = best_models[melhor_nome]
print(f"\n🏆 Melhor modelo: {melhor_nome} (F1={resultados[melhor_nome]['F1']:.4f})")

# ── 8. Exportação para test_requirements/ ────────────────────────────────────
model_path = os.path.join(OUTPUT_DIR, "model.pkl")
meta_path  = os.path.join(OUTPUT_DIR, "model_metadata.pkl")

with open(model_path, "wb") as f:
    pickle.dump(melhor_modelo, f)

metadata = {
    "teams":           sorted(df["Team"].unique().tolist()),
    "weathers":        sorted(df["Weather"].unique().tolist()),
    "tire_strategies": sorted(df["TireStrategy"].unique().tolist()),
    "features_num":    FEATURES_NUM,
    "features_cat":    FEATURES_CAT,
    "best_model":      melhor_nome,
    "metrics":         resultados[melhor_nome]
}

with open(meta_path, "wb") as f:
    pickle.dump(metadata, f)

print(f"\n💾 Artefatos exportados para test_requirements/")
print(f"   → model.pkl")
print(f"   → model_metadata.pkl")