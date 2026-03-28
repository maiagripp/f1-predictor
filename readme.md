# 🏎️ Previsão de Vencedor de Corrida de F1

Aplicação de Machine Learning para previsão de vencedor de corridas de Fórmula 1.
Dado o perfil de largada de um piloto, o modelo classifica se ele vai **vencer a corrida (P1) ou não**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maiagripp/f1-predictor/blob/main/notebook/f1_model_training.ipynb)

---

## 📁 Estrutura do Projeto

```
f1-predictor/
├── f1_enhanced_dataset_for_analysis.csv ← dataset F1
├── notebook/
│ └── f1_winner_predictor.ipynb ← notebook Colab (ML completo)
├── model/
│ └── train_model.py ← treina e exporta os .pkl (VSCode)
├── test_requirements/
│ ├── model.pkl ← gerado pelo train_model.py
│ └── model_metadata.pkl ← gerado pelo train_model.py
├── backend/
│ ├── app.py ← API Flask
│ └── requirements.txt
├── frontend/
│ ├── index.html ← interface web
│ └── style.css ← estilos
└── tests/
└── test_model.py ← testes automatizados PyTest
```

---

## ⚙️ Pré-requisitos

- Python 3.10+
- pip
- Navegador moderno (Chrome, Firefox, Edge)
- (Opcional) VSCode com extensão Live Server

---

## 🚀 Passo a Passo para Rodar o Projeto

### 1. Clonar o repositório

```bash
git clone https://github.com/maiagripp/f1-predictor.git
cd f1-predictor
```

---

### 2. Criar e ativar ambiente virtual (recomendado)

```bash
# Criar
python -m venv f1predictor

# Ativar — macOS/Linux
source f1predictor/bin/activate

# Ativar — Windows
f1predictor\Scripts\activate
```

---

### 3. Instalar dependências

```bash
pip install scikit-learn imbalanced-learn pandas numpy flask flask-cors pytest
```

---

### 4. Treinar o modelo (gera os arquivos .pkl)

> Certifique-se de que o arquivo `f1_enhanced_dataset_for_analysis.csv` está na **raiz do projeto**.

```bash
python model/train_model.py
```

Após a execução, os seguintes arquivos serão criados automaticamente:

```
test_requirements/
├── model.pkl
└── model_metadata.pkl
```

---

### 5. Iniciar o back-end (API Flask)

```bash
cd backend
python app.py
```

O servidor sobe em: **http://localhost:5000**

Para verificar se está rodando, acesse no browser ou via curl:

```bash
curl http://localhost:5000/
# {"status": "ok", "model": "SVM", ...}
```

---

### 6. Abrir o front-end

Abra o arquivo `frontend/index.html` diretamente no browser **ou** use o Live Server do VSCode:

1. Clique com o botão direito em `frontend/index.html`
2. Selecione **"Open with Live Server"**

> ⚠️ O back-end precisa estar rodando (passo 5) para o front-end funcionar.

---

### 7. Rodar os testes automatizados (PyTest)

```bash
pytest tests/test_model.py -v
```

Saída esperada:

`code
tests/test_model.py::test_model_loads PASSED
tests/test_model.py::test_model_f1 PASSED
tests/test_model.py::test_model_recall PASSED
tests/test_model.py::test_model_auc_roc PASSED
tests/test_model.py::test_model_precision PASSED
tests/test_model.py::test_output_shape PASSED
tests/test_model.py::test_binary_classes PASSED
tests/test_model.py::test_proba_sums_to_one PASSED
tests/test_model.py::test_pole_position_wins_more PASSED
`

---

## 📓 Notebook Google Colab

O notebook pode ser executado **de forma independente** no Google Colab, sem necessidade de configuração local.

1. Clique no badge **Open in Colab** no topo deste README
2. Faça upload do `f1_enhanced_dataset_for_analysis.csv` para o seu GitHub
3. Substitua `DATA_URL` no notebook pela URL raw do seu arquivo:

https://raw.githubusercontent.com/SEU_USUARIO/SEU_REPO/main/f1_enhanced_dataset_for_analysis.csv

4. Execute todas as células (`Runtime → Run all`)
5. Ao final, baixe `model.pkl` e `model_metadata.pkl` gerados
6. Coloque os arquivos em `test_requirements/` para usar no back-end

---

## 🔌 Endpoints da API

| Método | Rota        | Descrição                          |
|--------|-------------|-------------------------------------|
| GET    | `/`         | Health check e info do modelo       |
| GET    | `/metadata` | Lista equipes, climas e estratégias |
| POST   | `/predict`  | Realiza a predição                  |

### Exemplo de requisição POST `/predict`

```bash
curl -X POST http://localhost:5000/predict \
-H "Content-Type: application/json" \
-d '{
 "QualifyingPosition": 1,
 "StartPosition": 1,
 "PitStopCount": 2,
 "Weather": "Sunny",
 "TireStrategy": "Soft-Hard",
 "Team": "Red Bull Racing"
}'
```

### Resposta

```json
{
"prediction": 1,
"label": "🏆 Vencedor Previsto!",
"probability": {
 "nao_venceu": 0.2341,
 "venceu": 0.7659
}
}
```

---

## 🧠 Sobre o Modelo

| Algoritmos avaliados | KNN, Decision Tree, Naive Bayes, SVM |
|---|---|
| Target | `winner` (1 = P1, 0 = demais posições) |
| Features | QualifyingPosition, StartPosition, PitStopCount, Weather, TireStrategy, Team |
| Desbalanceamento | ~5% positivos → SMOTE + `class_weight='balanced'` |
| Métrica principal | F1-Score (mais adequada que accuracy para classes desbalanceadas) |
| Otimização | GridSearchCV com 5-fold cross-validation |

### Thresholds dos testes (PyTest)

| Métrica  | Threshold mínimo |
|----------|-----------------|
| F1-Score | ≥ 0.40          |
| Recall   | ≥ 0.35          |
| AUC-ROC  | ≥ 0.75          |
| Precision| ≥ 0.30          |

---

## 🛡️ Segurança e Boas Práticas

- Dados de saída nunca retornam os dados de entrada (evita exposição desnecessária)
- Em produção, proteger `/predict` com autenticação JWT
- Aplicar rate limiting para evitar uso abusivo
- Dataset usa dados públicos de F1 — para domínios com dados sensíveis, aplicar pseudonimização e conformidade com LGPD

---

## 📋 Dependências principais

flask==3.0.0
flask-cors==4.0.0
scikit-learn==1.4.0
imbalanced-learn==0.12.0
numpy==1.26.4
pandas==2.2.0
pytest


---

## 📄 Licença

MIT