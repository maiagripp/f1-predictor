"""
app.py — F1 Winner Predictor API
Carrega artefatos de ../test_requirements/ (gerados pelo train_model.py)
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# Caminho relativo: backend/ → ../test_requirements/
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS  = os.path.join(BASE_DIR, "..", "test_requirements")
MODEL_PATH = os.path.join(ARTIFACTS, "model.pkl")
META_PATH  = os.path.join(ARTIFACTS, "model_metadata.pkl")


def load_artifacts():
    """Carrega modelo e metadados de test_requirements/."""
    assert os.path.exists(MODEL_PATH), (
        f"model.pkl não encontrado em {MODEL_PATH}. "
        "Execute python model/train_model.py primeiro."
    )
    assert os.path.exists(META_PATH), (
        f"model_metadata.pkl não encontrado em {META_PATH}. "
        "Execute python model/train_model.py primeiro."
    )
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
    return model, metadata


model, metadata = load_artifacts()
print(f"✅ Modelo carregado: {metadata.get('best_model', 'desconhecido')}")


@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": metadata.get("best_model"),
        "metrics": metadata.get("metrics")
    })


@app.route("/metadata", methods=["GET"])
def get_metadata():
    return jsonify({
        "teams":           metadata["teams"],
        "weathers":        metadata["weathers"],
        "tire_strategies": metadata["tire_strategies"]
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    required = ["QualifyingPosition", "StartPosition", "PitStopCount",
                "Weather", "TireStrategy", "Team"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Campos ausentes: {missing}"}), 400

    try:
        input_df = pd.DataFrame([{
            "QualifyingPosition": float(data["QualifyingPosition"]),
            "StartPosition":      float(data["StartPosition"]),
            "PitStopCount":       float(data["PitStopCount"]),
            "Weather":            str(data["Weather"]),
            "TireStrategy":       str(data["TireStrategy"]),
            "Team":               str(data["Team"])
        }])
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Valor inválido: {e}"}), 400

    prediction = int(model.predict(input_df)[0])
    proba      = model.predict_proba(input_df)[0].tolist()
    label      = "🏆 Vencedor Previsto!" if prediction == 1 else "Sem vitória prevista"

    return jsonify({
        "prediction": prediction,
        "label": label,
        "probability": {
            "nao_venceu": round(proba[0], 4),
            "venceu":     round(proba[1], 4)
        }
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)