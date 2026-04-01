from fastapi import FastAPI
import joblib
import pandas as pd

from src.features import create_features

app = FastAPI()

# carregar modelo
model = joblib.load("models/model.pkl")

@app.get("/")
def home():
    return {"message": "API de risco de crédito rodando"}

@app.post("/predict")
def predict(data: dict):
    try:
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