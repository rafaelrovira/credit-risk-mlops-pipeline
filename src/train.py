import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
import joblib
from features import create_features
import os
import sklearn
import sys

# CONFIGURAÇÃO MLFLOW
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("credit-risk")

# CARREGAR DADOS
df = pd.read_csv("data/raw.csv")

# FEATURE ENGINEERING
df = create_features(df)

X = df[["income", "debt", "age", "debt_to_income", "is_young"]]
y = df["default"]

# SPLIT TREINO / TESTE
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42  # garante reprodutibilidade
)

# MODELO
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# AVALIAÇÃO
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print(f"Accuracy: {acc}")
print(classification_report(y_test, preds))

# MLFLOW
with mlflow.start_run():
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("sklearn_version", sklearn.__version__)
    mlflow.log_param("python_version", sys.version)

    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="credit-risk"
    )


    from mlflow.tracking import MlflowClient

    client = MlflowClient()

    latest_versions = client.get_latest_versions("credit-risk")
    latest_version = latest_versions[0].version

    client.transition_model_version_stage(
        name="credit-risk",
        version=latest_version,
        stage="Production"
    )

# SALVAR MODELO LOCAL 

joblib.dump(model, "models/model.pkl")
print("Modelo treinado e salvo com sucesso!")