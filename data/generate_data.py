import pandas as pd
import numpy as np

np.random.seed(42)

n = 1000

data = pd.DataFrame({
    "income": np.random.randint(1000, 10000, n),
    "debt": np.random.randint(0, 5000, n),
    "age": np.random.randint(18, 70, n),
})

# regra simples de risco
data["default"] = (
    (data["debt"] / data["income"] > 0.5) |
    (data["age"] < 25)
).astype(int)

# desbalanceamento proposital (pra verificar se o modelo mantém um equilibrio na acurácia)
data = data.sample(frac=1).reset_index(drop=True)
data = data.head(int(len(data) * 0.7))  

data.to_csv("data/raw.csv", index=False)

print("Dataset gerado!")