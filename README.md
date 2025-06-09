# Clasificacion_Alpha_Betha
Proyecto de clasificación, este proyecto consume el modelo de clasificación, recibe un json de una solicitud post, realiza la clasificación y entregar la respuesta. 

Esta API permite realizar predicciones utilizando un modelo de machine learning (XGBoost) previamente entrenado. A continuación, se detalla su funcionamiento básico:

### Características principales
* Endpoints disponibles:

/predict: Recibe datos en formato JSON y devuelve una predicción.

### Preprocesamiento automático:

* Codificación one-hot para variables categóricas.
* Estandarización de características.
* Selección de variables relevantes.

### Respuesta en JSON:

Devuelve la clase predicha en formato legible.

## Cómo usarla
### Ejemplo de solicitud (POST)
json
POST /predict  
{  
    "SeniorCity": 1,  
    "Partner": "Yes",  
    "Dependents": "No",  
    "Service1": "Fiber optic",  
    "Service2": "No",  
    "Security": "No",  
    "OnlineBackup": "No",  
    "DeviceProtection": "No",  
    "TechSupport": "No",  
    "Contract": "Month-to-month",  
    "PaperlessBilling": "Yes",  
    "PaymentMethod": "Electronic check",  
    "Charges": 70.5,  
    "Demand": 0.8  
}  
### Ejemplo de respuesta
json
{  
    "prediccion": "Betha"  
}  
### Requisitos
* Python 3.8+
* FastAPI
* Scikit-learn
* XGBoost
* Pandas

