import os
import random
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras
from tensorflow.keras import layers

os.makedirs("models", exist_ok=True)
random.seed(42)
np.random.seed(42)

# 1. Cargar los 812 tickets reales en español del dataset multilingüe
df_multi = pd.read_csv("data/dataset-tickets-multi-lang3-4k.csv")
es_df = df_multi[df_multi["language"] == "es"].copy()

priority_map = {"low": 0, "medium": 1, "high": 2}
type_map = {
    "Request": 0,   # Petición
    "Problem": 1,   # Queja
    "Incident": 2,  # Reclamo
    "Change": 3,    # Sugerencia
}
queue_map = {
    "Product Support": 0,                  # Servicio de Buses
    "Billing and Payments": 1,             # Tarifas y Pagos
    "Service Outages and Maintenance": 2,  # Horarios
    "Technical Support": 3,                # Rutas
    "IT Support": 4,                       # Infraestructura
    "Customer Service": 5,                 # Atención al cliente
    "Returns and Exchanges": 6,            # Accesibilidad
    "Human Resources": 7,                  # Seguridad
    "General Inquiry": 8,                  # Otros
    "Sales and Pre-Sales": 8,              # Otros
}

real_es = pd.DataFrame({
    "text": es_df["subject"].fillna("") + " " + es_df["body"].fillna(""),
    "type": es_df["type"].map(type_map).fillna(0).astype(int),
    "category": es_df["queue"].map(queue_map).fillna(8).astype(int),
    "priority": es_df["priority"].map(priority_map),
})

# 2. Reforzar vocabulario específico de transporte público (TransCaribe)
transcaribe_base = {
    2: [  # ALTA
        ("Falla grave en frenos del bus", "El articulado casi choca porque no respondieron los frenos", 2, 0),
        ("Robo dentro de la estación", "Me hurtaron el celular y la billetera sin vigilancia", 1, 7),
        ("Accidente de tránsito con heridos", "El bus frenó bruscamente y varios pasajeros cayeron", 2, 7),
        ("Acoso y agresión en el articulado", "Un sujeto agredía pasajeros y el conductor no se detuvo", 1, 7),
        ("Bus echando humo y olor a quemado", "El motor botaba humo negro y tocó evacuar de emergencia", 2, 0),
        ("Puertas abiertas con el bus en marcha", "Arrancó con las puertas abiertas poniendo vidas en riesgo", 2, 7),
        ("Exceso de velocidad y semáforos en rojo", "Conductor manejando de forma peligrosa en la troncal", 1, 7),
        ("Atraco en paradero alimentador", "Falta total de seguridad, atracaron a varios usuarios", 1, 7),
        ("Bus varado bloqueando carril solo bus", "Se quedó sin dirección bloqueando toda la troncal", 2, 0),
    ],
    1: [  # MEDIA
        ("Cobro doble en la tarjeta", "Pasé la tarjeta una vez y descontaron dos pasajes", 2, 1),
        ("Demora excesiva en la ruta X101", "Más de 45 minutos esperando el bus y no pasa ninguno", 1, 2),
        ("Aire acondicionado dañado", "Bus completamente lleno y sin aire funcionando", 1, 0),
        ("Rampa para silla de ruedas dañada", "La plataforma de accesibilidad no bajó al abordar", 2, 6),
        ("Mal trato del taquillero", "El funcionario respondió de forma grosera al recargar", 1, 5),
        ("No cargó el saldo en el punto", "Pagué la recarga en efectivo y no aparece el saldo", 2, 1),
        ("Bus no se detuvo en el paradero", "Le hice la parada con tiempo y siguió de largo", 1, 3),
    ],
    0: [  # BAJA
        ("Muy caro", "Considero que el valor del pasaje ha subido mucho", 0, 1),
        ("Solicitud de información de horarios", "A qué hora sale el primer bus los domingos", 0, 2),
        ("Felicitaciones al conductor", "Excelente servicio y amabilidad del operador", 4, 5),
        ("Sugerencia de nueva parada", "Sería bueno agregar un paradero cerca al centro comercial", 3, 3),
        ("Consulta puntos de recarga", "Dónde puedo adquirir tarjeta personalizada en Bocagrande", 0, 1),
        ("Felicitaciones a transcaperrito", "Me encantó la campaña educativa en estaciones", 4, 8),
    ],
}

tc_rows = []
for prioridad, items in transcaribe_base.items():
    for _ in range(40):
        asunto, desc, tipo, cat = random.choice(items)
        tc_rows.append({"text": f"{asunto} {desc}", "type": tipo, "category": cat, "priority": prioridad})

df_final = pd.concat([real_es, pd.DataFrame(tc_rows)], ignore_index=True)

# 3. Entrenar Red Neuronal con TF-IDF + Metadatos
X_train, X_test, y_train, y_test = train_test_split(
    df_final[["text", "type", "category"]],
    df_final["priority"],
    test_size=0.2,
    random_state=42,
    stratify=df_final["priority"],
)

preprocessor = ColumnTransformer([
    ("tfidf", TfidfVectorizer(max_features=2500, ngram_range=(1, 2), strip_accents="unicode"), "text"),
    ("meta", OneHotEncoder(handle_unknown="ignore"), ["type", "category"]),
])

X_train_vec = preprocessor.fit_transform(X_train).toarray()
X_test_vec = preprocessor.transform(X_test).toarray()

model = keras.Sequential([
    layers.Input(shape=(X_train_vec.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(32, activation="relu"),
    layers.Dense(3, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(X_train_vec, y_train, validation_split=0.2, epochs=15, batch_size=32, verbose=1)

y_pred = model.predict(X_test_vec, verbose=0).argmax(axis=1)
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=["BAJA", "MEDIA", "ALTA"]))

model.save("models/quejapp_priority_nn.keras")
joblib.dump(preprocessor, "models/priority_preprocessor.joblib")
print("Modelo entrenado con tickets reales en español + dominio TransCaribe.")