from __future__ import annotations
from typing import Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from Recommendation.recommender import Trainer


app = FastAPI()
trainer = Trainer('df_unpivoted.csv')

class QueryText(BaseModel):
    text: str

class StatusObject(BaseModel):
    status: str
    timestamp: str
    precision:float

class PredictionObject(BaseModel):
    text: str
    predictions: Dict

class PredictionsObject(BaseModel):
    predictions: List[PredictionObject]

@app.get("/")
def home():
    return({"message": "System is up"})

@app.get("/status", summary="Get current status of the system")
def get_status():
    status = trainer.get_status()
    return StatusObject(**status)

@app.get("/predict", summary="Predict single input")
def predict(query_text: QueryText):
    try:
        prediction = trainer.predict()
        # return PredictionObject(**prediction)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

