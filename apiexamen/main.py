from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from prophet import Prophet

app = FastAPI()

# Activer CORS pour React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Autoriser le frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modèle pour recevoir les données
class WeeklySalesData(BaseModel):
    values: list[dict]

@app.post("/forecast")
def predict_sales(data: WeeklySalesData):
    # Convertir en DataFrame
    df = pd.DataFrame(data.values)

    # Renommer les colonnes pour Prophet
    df = df.rename(columns={
        "[Due Date].[yqmw].[Week].[MEMBER_CAPTION]": "week",
        "[Measures].[LineTotal-Sales]": "sales"
    })

    # Convertir la colonne 'week' en date
    df["ds"] = pd.to_datetime(df["week"].str.replace("W/C ", ""), format="%d/%m/%y")
    df = df[["ds", "sales"]].rename(columns={"sales": "y"})  # Prophet attend 'ds' et 'y'

    # Agréger les données historiques par trimestre
    df["quarter"] = df["ds"].dt.to_period("Q")
    quarterly_sales = df.groupby("quarter")["y"].sum().reset_index()

    # Entraîner le modèle avec les valeurs hebdomadaires
    model = Prophet()
    model.fit(df)

    # Prédire pour les 4 prochains trimestres
    future = model.make_future_dataframe(periods=4, freq='Q')
    forecast = model.predict(future)

    # Résultat
    result = {
        "historical": quarterly_sales.to_dict(orient="records"),
        "quarterly_forecast": forecast[["ds", "yhat"]].set_index("ds").resample('Q').sum().reset_index().to_dict(orient="records"),
    }

    return result
