# 🏦 Credit Risk MLOps Pipeline

> Pipeline end-to-end de Machine Learning Engineering para predição de risco de crédito — do treinamento ao deploy e monitoramento em produção.

---

## 📌 Visão Geral

Este projeto simula um pipeline completo de **MLOps aplicado a crédito**, cobrindo todas as etapas de um ciclo de vida real de modelo em produção:

| Etapa | Descrição |
|---|---|
| 🔧 Feature Engineering | Criação e transformação de variáveis preditivas |
| 🤖 Treinamento | Modelagem com rastreamento de experimentos via MLflow |
| 🚀 Deploy | Exposição do modelo via API REST com FastAPI |
| 📊 Monitoramento | Acompanhamento básico de métricas em produção |
| ✅ Boas Práticas | Consistência treino/inferência, versionamento e reprodutibilidade |

---

## 🏗️ Arquitetura

```
Dados Brutos
    │
    ▼
Feature Engineering  ──►  Consistência entre treino e inferência
    │
    ▼
Treinamento do Modelo  ──►  MLflow (rastreamento de experimentos)
    │
    ▼
Artefato do Modelo (.pkl)
    │
    ▼
API FastAPI  ──►  POST /predict
    │
    ▼
Predição de Risco de Crédito
```

---

## 📁 Estrutura do Projeto

```
mlops-credit-risk/
│
├── data/               # Dados brutos e scripts de geração de dataset
├── src/                # Pipeline de treino e feature engineering
├── api/                # API FastAPI para inferência em tempo real
├── models/             # Modelos treinados (não versionados no git)
└── notebooks/          # Exploração e análise exploratória (EDA)
```

---

## 🛠️ Tecnologias

| Ferramenta | Finalidade |
|---|---|
| **Python** | Linguagem principal |
| **scikit-learn** | Modelagem e pré-processamento |
| **FastAPI** | Serving do modelo via REST API |
| **MLflow** | Rastreamento de experimentos e registro de modelos |

---

## ⚙️ Feature Engineering

As seguintes variáveis derivadas são criadas e aplicadas de forma consistente tanto no treino quanto na inferência:

| Feature | Descrição |
|---|---|
| `debt_to_income` | Razão entre dívida e renda — indicador clássico de alavancagem |
| `is_young` | Flag binária indicando clientes jovens (maior risco histórico) |

> ⚠️ **Importante:** A mesma lógica de feature engineering é aplicada no pipeline de treino e na API, garantindo consistência e evitando *training-serving skew*.

---

## 🤖 Modelo

- **Algoritmo:** Logistic Regression
- **Framework:** scikit-learn

### Métricas de Avaliação

| Métrica | Descrição |
|---|---|
| Accuracy | Proporção de predições corretas |
| Precision | Dos preditos como default, quantos realmente são |
| Recall | Dos inadimplentes reais, quantos foram capturados |
| F1-Score | Equilíbrio entre Precision e Recall |
| AUC-ROC | Capacidade discriminativa geral do modelo |

---

## 🚀 API — Inferência em Tempo Real

A API expõe um endpoint REST para predição individual de risco de crédito.

### Endpoint

```
POST /predict
```

### Exemplo de Request

```json
{
  "income": 3000,
  "debt": 2000,
  "age": 22
}
```

### Exemplo de Response

```json
{
  "default_risk": 0.99,
  "prediction": 1,
  "threshold": 0.5
}
```

| Campo | Tipo | Descrição |
|---|---|---|
| `default_risk` | float | Probabilidade estimada de inadimplência (0 a 1) |
| `prediction` | int | Classificação binária: 1 = alto risco, 0 = baixo risco |
| `threshold` | float | Limiar de decisão utilizado |

---

## 🚦 Como Executar

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Treinar o modelo

```bash
python src/train.py
```

### 3. Subir a API

```bash
uvicorn api.main:app --reload
```

### 4. Fazer uma predição

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"income": 3000, "debt": 2000, "age": 22}'
```

---

## 📈 MLflow — Rastreamento de Experimentos

Com o MLflow ativo, é possível visualizar experimentos, métricas e artefatos:

```bash
mlflow ui
```

Acesse em: `http://localhost:5000`

---

## 🔮 Próximos Passos

- [ ] Adicionar testes unitários e de integração
- [ ] Containerizar com Docker
- [ ] Implementar CI/CD com GitHub Actions
- [ ] Adicionar monitoramento de data drift
- [ ] Deploy em cloud (AWS / GCP / Azure)

---

## 📄 Licença

Projeto para fins educacionais e de portfólio.
