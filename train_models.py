import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


# 1. Ler o dataset
dados = pd.read_csv("dataset.csv")


# 2. Separar entradas e respostas
X = dados.drop("label", axis=1)
y = dados["label"]


# 3. Separar dados de treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 4. Criar o modelo
modelo = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)


# 5. Treinar
modelo.fit(X_treino, y_treino)


# 6. Testar
previsoes = modelo.predict(X_teste)

acuracia = accuracy_score(
    y_teste,
    previsoes
)

print("Acurácia:", acuracia)


# 7. Salvar o modelo treinado
joblib.dump(
    modelo,
    "models/modelo_libras.pkl"
)

print("Modelo salvo!")