import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import joblib

from features import create_features

# carregar dados
df = pd.read_csv("data/raw.csv")

# features
df = create_features(df)

X = df[["income", "debt", "age", "debt_to_income", "is_young"]]
y = df["default"]

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# modelo
model = LogisticRegression()
model.fit(X_train, y_train)

# avaliação
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print(f"Accuracy: {acc}")
print(classification_report(y_test, preds))

# salvar modelo
joblib.dump(model, "models/model.pkl")