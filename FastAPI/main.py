from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np

# Inicializar FastAPI
app = FastAPI()

# Cargar modelo y objetos de preprocesamiento
model = joblib.load("modelo_clasificacion_xgb.pkl")
scaler = joblib.load("scaler.pkl")
selector = joblib.load("selector.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Columnas de entrada esperadas
INPUT_COLUMNS = [
    "SeniorCity", "Partner", "Dependents", "Service1", "Service2",
    "Security", "OnlineBackup", "DeviceProtection", "TechSupport",
    "Contract", "PaperlessBilling", "PaymentMethod", "Charges", "Demand"
]

# Definir el esquema del JSON de entrada
class InputData(BaseModel):
    SeniorCity: int
    Partner: str
    Dependents: str
    Service1: str
    Service2: str
    Security: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    Charges: float
    Demand: float

@app.post("/predict")
def predict(data: InputData):
    # Convertir a DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Asegurar consistencia de tipos
    cat_cols = input_df.select_dtypes(include=["object"]).columns.tolist()
    input_df[cat_cols] = input_df[cat_cols].astype(str)

    # One-hot encoding igual que en entrenamiento
    input_encoded = pd.get_dummies(input_df)

    # Alinear con columnas del entrenamiento (pueden faltar columnas nuevas)
    all_cols = scaler.feature_names_in_  # columnas usadas en el fit
    for col in all_cols:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[all_cols]  # reordenar

    # Escalar
    X_scaled = scaler.transform(input_encoded)

    # Selección de variables
    X_sel = selector.transform(X_scaled)

    # Predicción
    pred = model.predict(X_sel)
    pred_label = label_encoder.inverse_transform(pred)[0]

    return {"prediccion": pred_label}
