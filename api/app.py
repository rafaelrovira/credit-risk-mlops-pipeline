from fastapi import FastAPI
import joblib
import pandas as pd
import os
import mlflow.sklearn

from src.features import create_features

app = FastAPI()

# carregar modelo
MODEL_URI = os.getenv("MODEL_URI", "models:/credit-risk/Production")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "model.pkl")
model = None  # Carregamento lazy

def load_model():
    global model
    if model is None:
        try:
            print(f"[LOG] Carregando modelo de {MODEL_URI}")
            model = mlflow.sklearn.load_model(MODEL_URI)
            print("[LOG] Modelo carregado com sucesso!")
        except Exception as e:
            print(f"[ERRO] Falha ao carregar modelo de {MODEL_URI}: {str(e)}")
            print("[LOG] Tentando carregar modelo local...")
            try:
                model = joblib.load(model_path)
                print("[LOG] Modelo local carregado com sucesso!")
            except Exception as e2:
                print(f"[ERRO] Falha ao carregar modelo local: {str(e2)}")
                raise RuntimeError(f"Não foi possível carregar o modelo: {str(e2)}")

@app.get("/")
def home():
    return {"message": "API de risco de crédito rodando"}

@app.post("/predict")
def predict(data: dict):
    try:
        load_model()  # Carregar modelo se não estiver carregado
        print(f"[LOG] Input recebido: {data}")  # dinheiro recebido
        
        df = pd.DataFrame([data])
        df = create_features(df)
        
        X = df[["income", "debt", "age", "debt_to_income", "is_young"]]
        
        prob = model.predict_proba(X)[0][1]
        
        print(f"[LOG] Probabilidade gerada: {prob}") # probabilidade gerada, quanto maior, maior o risco de inadimplencia
        
        return {
            "default_risk": float(prob),
            "prediction": int(prob > 0.5),
            "threshold": 0.5
        }
    
    except Exception as e:
        print(f"[ERRO] {str(e)}")  #erro ocorrido
        return {"error": str(e)}