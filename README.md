# Credit Risk MLOps Pipeline

Projeto end-to-end de Machine Learning Engineering focado em risco de crédito, cobrindo desde o treinamento até deploy e monitoramento de modelo.

---

## Objetivo

Simular um pipeline completo de MLOps, incluindo:

- Feature engineering
- Treinamento de modelo
- Deploy via API
- Monitoramento básico
- Boas práticas de produção

---

## Arquitetura

Dados → Feature Engineering → Treinamento → Modelo → API → Predição

---

## Estrutura do Projeto

mlops-credit-risk/
    │
    ├── data/ # Dados brutos e geração de dataset
    ├── src/ # Pipeline de treino e features
    ├── api/ # API FastAPI para inferência
    ├── models/ # Modelos treinados (não versionados no git)
    ├── notebooks/ # Exploração (opcional)

---

## Tecnologias

- Python
- scikit-learn
- FastAPI
- MLflow

---

# Feature Engineering
Criação de variáveis derivadas:

- debt_to_income
- is_young

Garantindo consistência entre treino e inferência.

---

## Modelo

- Logistic Regression
- Métricas avaliadas:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - AUC

---

## API (Inferência em tempo real)

Endpoint:

    Exemplo de input:

```json
{
  "income": 3000,
  "debt": 2000,
  "age": 22
}

Resposta :

{
  "default_risk": 0.99,
  "prediction": 1,
  "threshold": 0.5
}