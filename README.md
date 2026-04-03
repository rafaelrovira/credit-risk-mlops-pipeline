# Credit Risk MLOps Pipeline

> Pipeline end-to-end de Machine Learning Engineering para predição de risco de crédito — do treinamento ao deploy e monitoramento em produção.

---

## Visão Geral

Este projeto simula um pipeline completo de **MLOps aplicado a crédito**, cobrindo todas as etapas de um ciclo de vida real de modelo em produção:

| Etapa | Descrição |
|---|---|
| Feature Engineering | Criação e transformação de variáveis preditivas |
| Treinamento | Modelagem com rastreamento de experimentos via MLflow |
| Deploy | Exposição do modelo via API REST com FastAPI |
| Monitoramento | Acompanhamento básico de métricas em produção |
| Boas Práticas | Consistência treino/inferência, versionamento e reprodutibilidade |

---

## Arquitetura

```
                        ┌─────────────────┐
                        │   Dados Brutos  │
                        └────────┬────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────┐
│                    DOCKER CONTAINER: train                     │
│  Feature Engineering → Treinamento → Registro no MLflow        │
└────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  MLflow Server  │ (Container: mlflow)
                        │  Model Registry │ (Porta: 5000)
                        └────────┬────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────┐
│               DOCKER CONTAINER: inference_api                  │
│     FastAPI → Load Model → Feature Eng → Predict               │
│                  (Porta: 8000/predict)                         │
└────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────┐
                    │ Predição de Risco  │
                    │ {default_risk}     │
                    └────────────────────┘
```

### Networking Docker

```
┌─────────────────────────────────────────────────┐
│    Docker Network: credit-risk-mlops-pipeline   │
├─────────────────────────────────────────────────┤
│                                                 │
│  mlflow        — http://mlflow:5000             │
│  train_pipeline — http://train_pipeline         │
│  inference_api — http://0.0.0.0:8000 (externa) │
│                                                 │
└─────────────────────────────────────────────────┘
```

Os containers se comunicam via DNS interno da rede Docker. A API usa a variável de ambiente `MLFLOW_TRACKING_URI=http://mlflow:5000` para conectar ao servidor MLflow.

---

## Estrutura do Projeto

```
credit-risk-mlops-pipeline/
│
├── Dockerfile                  # Imagem base para pipeline de treino
├── docker-compose.yaml         # Orquestração dos containers (MLflow, API, Train)
├── requirements.txt            # Dependências Python
├── README.md                   # Este arquivo
│
├── data/                       # Dados brutos e scripts de geração
│   ├── raw.csv                # Dataset de exemplo
│   └── generate_data.py        # Script para gerar dados de teste
│
├── src/                        # Pipeline de treino e feature engineering
│   ├── train.py               # Script de treinamento (integrado com MLflow)
│   └── features.py            # Lógica de feature engineering
│
├── api/                        # API FastAPI para inferência
│   ├── Dockerfile             # Imagem específica para a API
│   ├── __init__.py
│   └── app.py                 # Aplicação FastAPI com endpoint /predict
│
├── models/                     # Artefatos de modelos treinados
│   └── model.pkl              # Modelo serializado (gitignored)
│
└── mlruns/                     # Artifacts do MLflow (gitignored)
    └── [experimentos e runs]
```


---

## Tecnologias

| Ferramenta | Finalidade |
|---|---|
| **Python 3.11** | Linguagem principal |
| **scikit-learn** | Modelagem e pré-processamento |
| **FastAPI** | Serving do modelo via REST API |
| **MLflow** | Rastreamento de experimentos e registro de modelos |
| **Docker** | Containerização para ambiente reprodutível |
| **Docker Compose** | Orquestração de múltiplos containers em produção |
| **Uvicorn** | ASGI server para servir a API FastAPI |

---

## Infraestrutura — Containerização para Produção

Este projeto utiliza **Docker** e **Docker Compose** para replicar um ambiente de produção real com múltiplos serviços orquestrados:

### Serviços em Execução

| Container | Porta | Descrição | Responsabilidade |
|---|---|---|---|
| **mlflow** | 5000 | Tracking de experimentos e Model Registry | Armazenar metadata e modelos registrados |
| **train_pipeline** | — | Pipeline de treinamento em batch | Executar treinamento e registrar modelos no MLflow |
| **inference_api** | 8000 | API REST de predição | Servir predições em tempo real |

### Exemplo de docker-compose.yaml

```yaml
version: "3.9"

services:
  mlflow:
    image: python:3.11-slim
    container_name: mlflow
    command: >
      sh -c "pip install mlflow &&
          mlflow ui --host 0.0.0.0 --port 5000"
    ports:
      - "5000:5000"

  train:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: train_pipeline
    depends_on:
      - mlflow
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000

  api:
    build:
      context: .
      dockerfile: api/Dockerfile
    container_name: inference_api
    ports:
      - "8000:8000"
    depends_on:
      - mlflow
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
```

### Vantagens do Approach com Docker

- **Reprodutibilidade:** Mesmo ambiente em dev, test e produção  
- **Isolamento:** Cada serviço roda em seu próprio container  
- **Escalabilidade:** Fácil replicar containers ou orquestrar com Kubernetes  
- **Ciência Reprodutível:** Versionamento de imagens garante rastreabilidade  
- **Facilita Onboarding:** Novos desenvolvedores executam um único comando  

---

As seguintes variáveis derivadas são criadas e aplicadas de forma consistente tanto no treino quanto na inferência:

| Feature | Descrição |
|---|---|
| `debt_to_income` | Razão entre dívida e renda — indicador clássico de alavancagem |
| `is_young` | Flag binária indicando clientes jovens (maior risco histórico) |

> **Importante:** A mesma lógica de feature engineering é aplicada no pipeline de treino e na API, garantindo consistência e evitando *training-serving skew*.

---

## Modelo

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

## API — Inferência em Tempo Real

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

## Como Executar

### Opção 1: Com Docker (Recomendado para Produção)

#### Pré-requisitos
- Docker e Docker Compose instalados

#### Passos

```bash
# 1. Entrar no diretório do projeto
cd credit-risk-mlops-pipeline

# 2. Subir todos os serviços (MLflow, API, Pipeline de Treino)
docker-compose up -d

# 3. Verificar se os containers estão rodando
docker ps
```

#### Acessar os Serviços

| Serviço | URL |
|---|---|
| **API FastAPI** | http://localhost:8000 |
| **API Docs (Swagger)** | http://localhost:8000/docs |
| **MLflow UI** | http://localhost:5000 |

#### Fazer uma Predição via Docker

```bash
# Via curl
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"income": 3000, "debt": 2000, "age": 22}'

# Ou via Swagger em: http://localhost:8000/docs
```

#### Parar os Containers

```bash
docker-compose down
```

---

### Opção 2: Execução Local (Desenvolvimento)

#### Pré-requisitos
- Python 3.11+
- pip

#### Passos

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Iniciar MLflow (em um terminal separado)
mlflow ui

# 3. Treinar o modelo (em outro terminal)
python src/train.py

# 4. Subir a API (em outro terminal)
uvicorn api.app:app --reload
```

#### Acessar Localmente

- **API:** http://localhost:8000
- **Docs:** http://localhost:8000/docs
- **MLflow:** http://localhost:5000

#### Fazer uma Predição

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"income": 3000, "debt": 2000, "age": 22}'
```

---

## MLflow — Rastreamento de Experimentos

Com o MLflow ativo, é possível visualizar experimentos, métricas e artefatos:

```bash
mlflow ui
```

Acesse em: `http://localhost:5000`

---

## Troubleshooting

### API não sobe no Docker?

```bash
# Ver os logs detalhados
docker logs inference_api

# Reconstruir a imagem do zero
docker-compose down
docker-compose build --no-cache api
docker-compose up -d
```

### MLflow não encontra modelos registrados?

Certifique-se de que:
1. O MLflow está rodando e acessível da API (`http://mlflow:5000`)
2. O modelo foi treinado e registrado antes de iniciar a API
3. A variável de ambiente `MLFLOW_TRACKING_URI` está corretamente definida

### Porta já em uso?

Se a porta 8000 ou 5000 já está ocupada:

```bash
# Mudar a porta no docker-compose.yaml
# Exemplo: "9000:8000" usa porta 9000 localmente
```

---

## Production vs Development

| Aspecto | Development | Production (Docker) |
|---|---|---|
| **Isolamento** | Compartilhado com sistema | Containers isolados |
| **Reprodutibilidade** | Dependências de máquina local | Garantida (imagem fixa) |
| **Escalabilidade** | Vertical apenas | Horizontal (múltiplos containers) |
| **Logs** | stdout do terminal | Acessível via `docker logs` |
| **Configuração** | Variáveis de ambiente locais | Docker Compose `.env` + image |
| **Versionamento** | Tag de imagem = versão | Tagging automático recomendado |
| **Resiliência** | Sem restart automático | Restart policies via Compose |

### Exemplo de Tag de Produção

```bash
docker build -t credit-risk-api:v1.0.0 -f api/Dockerfile .
docker push seu-registry/credit-risk-api:v1.0.0
```

---

## Notas Importantes — Carregamento Lazy do Modelo

A API implementa **carregamento lazy (sob demanda) do modelo**:

```python
# Bom: A API inicia rapidamente
def load_model():
    global model
    if model is None:
        try:
            model = mlflow.sklearn.load_model(MODEL_URI)
        except Exception as e:
            # Fallback: tenta carregar modelo local
            model = joblib.load(model_path)
```

### Por que Lazy Loading?

- **A API inicia instantaneamente**, mesmo sem modelo disponível  
- **Fallback para modelo local** se MLflow não estiver acessível  
- **Evita timeout** durante inicialização do container  
- **Padrão recomendado** em produção (veremos erro na primeira requisição, não na startup)  

### Fluxo de Carregamento

```
1. docker-compose up          → API inicia sem modelo
2. Primeira /predict          → Carrega modelo automaticamente
3. Chamadas subsequentes      → Usa modelo em cache
```

---

Projeto para fins educacionais e de portfólio.
