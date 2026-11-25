from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from tensorflow import keras
from api_ia import router as tiempo_router

app = FastAPI(title="QuejAPP - API Redes Neuronales")

# Necesario para que Angular pueda consumir /predecir desde el navegador sin bloqueo CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
model = keras.models.load_model(BASE_DIR / "models" / "quejapp_priority_nn.keras")
preprocessor = joblib.load(BASE_DIR / "models" / "priority_preprocessor.joblib")
LABEL_MAP = {0: "BAJA", 1: "MEDIA", 2: "ALTA"}


class PQRSInput(BaseModel):
    # Ignora campos adicionales que envía Spring Boot (department, recievedDate, past30dTickets)
    model_config = ConfigDict(extra="ignore")
    subject: str | None = ""
    description: str | None = ""
    type: int | None = 0
    category: int | None = 0


@app.post("/predict-priority")
def predict_priority(data: PQRSInput):
    subject = data.subject or ""
    description = data.description or ""
    p_type = data.type if data.type is not None else 0
    category = data.category if data.category is not None else 0

    df_in = pd.DataFrame([{
        "text": f"{subject} {description}".strip(),
        "type": p_type,
        "category": category,
    }])

    X_vec = preprocessor.transform(df_in).toarray()
    probs = model.predict(X_vec, verbose=0)[0]
    idx = int(probs.argmax())

    return {
        "priority": LABEL_MAP[idx],
        "confidence": round(float(probs[idx]), 4),
        "probabilities": {
            "BAJA": round(float(probs[0]), 4),
            "MEDIA": round(float(probs[1]), 4),
            "ALTA": round(float(probs[2]), 4),
        },
    }


# Registra los endpoints de tiempo de respuesta (/ y /predecir)
app.include_router(tiempo_router)