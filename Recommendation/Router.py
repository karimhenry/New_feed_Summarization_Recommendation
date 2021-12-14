from __future__ import annotations
from typing import Dict, List, Optional, Union,DefaultDict
from enum import Enum

from fastapi import FastAPI, HTTPException, Body, Request, File, UploadFile, Form, Depends, BackgroundTasks

from pydantic import BaseModel
from Recommendation.surprise_recommender import SurpriseTrainer
from pydantic import parse_obj_as
from collections import defaultdict
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse


app = FastAPI()
trainer = SurpriseTrainer(1)
temps = Jinja2Templates(directory='templates')

# @app.get("/home/{user_name}", response_class=HTMLResponse)
# def write_home(request: Request, user_name: str):
#     return temps.TemplateResponse("test3.html", {"request": request, "username": user_name})

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
    # predictions: list

class User(BaseModel):
    name: list
    # age: list


# @app.get("/home/{user_name}", response_class=HTMLResponse)
# def write_home(request: Request, user_name: str):
#     return temps.TemplateResponse("test3.html", {"request": request, "username": user_name})

@app.get("/",response_class=HTMLResponse)
def home(request: Request):
    return temps.TemplateResponse("home.html",{"request": request})
    # return({"message": "System is up"})

@app.get("/status", summary="Get current status of the system")
def get_status():
    status = trainer.get_status()
    return StatusObject(**status)

@app.get("/predict", summary="Predict single input")
def predict(user_id,n):
    try:
        predictions = trainer.get_top_n(user_id,int(n))
        # print(type(predictions))
        return predictions

    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


# @app.get("/items/{id}", response_class=HTMLResponse)
# async def read_item(request: Request, id: str):
#     return temps.TemplateResponse("item.html", {"request": request, "id": id})




