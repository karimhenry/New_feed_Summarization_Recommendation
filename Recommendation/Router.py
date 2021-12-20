from __future__ import annotations
from typing import Dict, List
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from Recommendation.surprise_recommender import SurpriseTrainer
import os
import pandas as pd

trainer = SurpriseTrainer(0)
app = FastAPI()
temps = Jinja2Templates(directory='templates')
app.mount("/static", StaticFiles(directory="static"), name="static")


class QueryText(BaseModel):
    text: str


class StatusObject(BaseModel):
    status: str
    timestamp: str
    precision: float


class PredictionObject(BaseModel):
    text: str
    predictions: Dict


class PredictionsObject(BaseModel):
    predictions: List[PredictionObject]
    # predictions: list


class User(BaseModel):
    name: list
    # age: list


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    user_id = "New User"
    top = 5

    titles = []
    summaries = []
    urls = []

    # Loading News Dataset
    path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data'), 'raw')
    news_df = pd.read_csv(path + '/news.tsv', sep='\t', header=None,
                          names=['News ID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'Title Entities',
                                 'Abstract Entities']).fillna("")

    top_news = news_df["News ID"][:top]

    for article in top_news:
        titles.append(str(news_df[news_df['News ID'] == article].iloc[0]['Title']))
        summaries.append(str(news_df[news_df['News ID'] == article].iloc[0]['Abstract']))
        urls.append(str(news_df[news_df['News ID'] == article].iloc[0]['URL']))
    return temps.TemplateResponse("index.html",
                                  {"request": request, "user_id": user_id, "predictions": top_news,
                                   "titles": titles, "summaries": summaries, "urls": urls})


@app.get("/recommend", response_class=HTMLResponse)
def recommend(request: Request, user_id, n=5):
    try:
        titles = []
        summaries = []
        urls = []

        # Loading News Dataset
        path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data'), 'raw')
        news_df = pd.read_csv(path + '/news.tsv', sep='\t', header=None,
                              names=['News ID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'Title Entities',
                                     'Abstract Entities']).fillna("")

        predictions = trainer.get_top_n(user_id, int(n))

        if not predictions:
            # Use Cold Start Option
            top_news = news_df["News ID"][:n]

            for article in top_news:
                titles.append(str(news_df[news_df['News ID'] == article].iloc[0]['Title']))
                summaries.append(str(news_df[news_df['News ID'] == article].iloc[0]['Abstract']))
                urls.append(str(news_df[news_df['News ID'] == article].iloc[0]['URL']))
            return temps.TemplateResponse("index.html",
                                          {"request": request, "user_id": user_id, "predictions": top_news,
                                           "titles": titles, "summaries": summaries, "urls": urls})
        else:
            for article in predictions['Articles']:
                titles.append(str(news_df[news_df['News ID'] == article].iloc[0]['Title']))
                summaries.append(str(news_df[news_df['News ID'] == article].iloc[0]['Abstract']))
                urls.append(str(news_df[news_df['News ID'] == article].iloc[0]['URL']))

            return temps.TemplateResponse("index.html",
                                          {"request": request, "user_id": user_id, "predictions": predictions,
                                           "titles": titles, "summaries": summaries, "urls": urls})

    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/category", response_class=HTMLResponse)
def category(request: Request, category, n=5):
    try:
        titles = []
        summaries = []
        urls = []
        predictions = []

        # Loading News Dataset
        path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data'),
                            'processed')
        news_df = pd.read_csv(path + '/Category_df.csv').fillna("")
        news_df = news_df[news_df['Category'] == category.lower()]
        n = min(n, news_df.shape[0])

        for i in range(n):
            predictions.append(str(news_df.iloc[i]['News ID']))
            titles.append(str(news_df.iloc[i]['Title']))
            summaries.append(str(news_df.iloc[i]['Abstract']))
            urls.append(str(news_df.iloc[i]['URL']))

        return temps.TemplateResponse("index.html",
                                      {"request": request, "user_id": category, "predictions": predictions,
                                       "titles": titles, "summaries": summaries, "urls": urls})

    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/status", summary="Get current status of the system")
def get_status():
    status = trainer.get_status()
    return StatusObject(**status)


@app.get("/predict", summary="Predict single input")
def predict(user_id, n):
    try:
        predictions = trainer.get_top_n(user_id, int(n))
        return predictions

    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
