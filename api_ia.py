from datetime import datetime
from pathlib import Path
from typing import Union
from zoneinfo import ZoneInfo
import joblib
import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["Predicción de Tiempo de Respuesta"])

BASE_DIR = Path(__file__).resolve().parent
modelo = joblib.load(BASE_DIR / "models" / "modelo_quejapp_nn.pkl")

ETIQUETAS_LEGIBLES = {
    "MuyRapido": "Muy rápido",
    "Rapido": "Rápido",
    "Lento": "Lento",
    "MuyLento": "Muy lento",
}


class SolicitudPqr(BaseModel):
    type: Union[int, str]
    category: Union[int, str]
    subject: str
    description: str


@router.api_route("/", methods=["GET", "HEAD"])
@router.api_route("/health", methods=["GET", "HEAD"])
def estado():
    return {
        "estado": "API IA QuejAPP activa",
        "modelos": ["Prioridad (Keras)", "Tiempo de respuesta (MLPClassifier)"],
    }


@router.post("/predecir")
def predecir_tiempo(pqr: SolicitudPqr):
    ahora = datetime.now(ZoneInfo("America/Bogota"))
    texto_completo = f"{pqr.subject} {pqr.description}".strip()

    entrada = pd.DataFrame([{
        "type": pqr.type,
        "category": pqr.category,
        "texto_completo": texto_completo,
        "hora_radicacion": ahora.hour,
        "dia_semana": ahora.weekday(),
    }])

    prediccion = str(modelo.predict(entrada)[0])
    probabilidades = modelo.predict_proba(entrada)[0]
    clases = modelo.classes_

    detalle_probs = {
        str(clase): round(float(prob) * 100, 2)
        for clase, prob in zip(clases, probabilidades)
    }
    confianza = max(detalle_probs.values())

    return {
        "prediccion": prediccion,
        "prediccion_legible": ETIQUETAS_LEGIBLES.get(prediccion, prediccion),
        "confianza": confianza,
        "probabilidades": detalle_probs,
    }