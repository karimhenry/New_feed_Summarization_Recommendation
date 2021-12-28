from __future__ import annotations
from typing import Dict, List
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from Data_Preprocessing.DataProcessing import DataPreprocess
from Recommendation.surprise_recommender import SurpriseTrainer
from Recommendation.matrix_factorization_recommender import matrix_factorization

import os
import random
import pandas as pd

# Data PreProcessing
preprocessing = DataPreprocess()

# Recommender Models
trainer = matrix_factorization(0)
trainer = SurpriseTrainer(0)

# Web API
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


class User(BaseModel):
    name: list


@app.get("/", response_class=HTMLResponse)
def home(request: Request, user_id="New User", n=9):
    try:
        # Load Recommendations with Cold-Start
        predictions, titles, summaries, urls, categories = [], [], [], [], []

        # Loading News Dataset
        path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data'), 'processed')
        news_df = pd.read_csv(path + '/Category_df.csv').fillna("")
        news_df = news_df.sort_values(by=['rating'], ascending=False)

        # Loading Summary Dataset
        path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data'), 'summary')
        summary_df = pd.read_pickle(path + '/LexRankSummary.p')
        summary_df = summary_df[['News ID', 'Cleaned_Article', 'Summary (TextRank)', 'Summary (LexRank)']]

        # Merging News With Summary Dataset
        news_df = news_df.merge(summary_df, how='left', left_on='News ID', right_on='News ID').fillna("")
        news_df = news_df.replace("Nan", "").replace("nan", "")

        for i in range(n):
            predictions.append(str(news_df.iloc[i]['News ID']))
            categories.append(str(news_df.iloc[i]['Category']).capitalize())
            titles.append(str(news_df.iloc[i]['Title']))
            urls.append(str(news_df.iloc[i]['URL']))

            # (Summary of TextRank) Then (Abstract Summary from dataset) Then (Article) Then (Empty String)
            if news_df.iloc[i]['Summary (TextRank)'] != "":
                summaries.append(str(news_df.iloc[i]['Summary (TextRank)']).capitalize())
            elif news_df.iloc[i]['Abstract'] != "":
                summaries.append("Abstract Summary :- " + str(news_df.iloc[i]['Abstract']).capitalize())
            elif news_df.iloc[i]['Cleaned_Article'] != "":
                summaries.append("Article :- " + str(news_df.iloc[i]['Cleaned_Article']).capitalize())
            else:
                summaries.append("")

        test_user = random.choice(preprocessing.Users())

        return temps.TemplateResponse("index.html",
                                      {"request": request, "user_id": user_id, "predictions": predictions, "urls": urls,
                                       "categories": categories, "titles": titles, "summaries": summaries, "test_user": test_user})

    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/category", response_class=HTMLResponse)
def category(request: Request, category, n=9):
    try:
        predictions, titles, summaries, urls, categories = [], [], [], [], []

        # Loading News Dataset
        path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data'), 'processed')
        news_df = pd.read_csv(path + '/Category_df.csv').fillna("")
        news_df = news_df[news_df['Category'] == category.lower()]
        news_df = news_df.sort_values(by=['rating'], ascending=False)

        # Loading Summary Dataset
        path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data'), 'summary')
        summary_df = pd.read_pickle(path + '/LexRankSummary.p')
        summary_df = summary_df[['News ID', 'Cleaned_Article', 'Summary (TextRank)', 'Summary (LexRank)']]

        # Merging News With Summary Dataset
        news_df = news_df.merge(summary_df, how='left', left_on='News ID', right_on='News ID').fillna("")
        news_df = news_df.replace("Nan", "").replace("nan", "")

        # Looping on minimum size between "n" and "News Dataset rows"
        n = min(n, news_df.shape[0])

        for i in range(n):
            predictions.append(str(news_df.iloc[i]['News ID']))
            categories.append(str(news_df.iloc[i]['Category']).capitalize())
            titles.append(str(news_df.iloc[i]['Title']))
            urls.append(str(news_df.iloc[i]['URL']))

            # (Summary of TextRank) Then (Abstract Summary from dataset) Then (Article) Then (Empty String)
            if news_df.iloc[i]['Summary (TextRank)'] != "":
                summaries.append(str(news_df.iloc[i]['Summary (TextRank)']).capitalize())
            elif news_df.iloc[i]['Abstract'] != "":
                summaries.append("Abstract Summary :- " + str(news_df.iloc[i]['Abstract']).capitalize())
            elif news_df.iloc[i]['Cleaned_Article'] != "":
                summaries.append("Article :- " + str(news_df.iloc[i]['Cleaned_Article']).capitalize())
            else:
                summaries.append("")

        test_user = random.choice(preprocessing.Users())

        return temps.TemplateResponse("index.html",
                                      {"request": request, "user_id": category, "predictions": predictions, "urls": urls,
                                       "categories": categories, "titles": titles, "summaries": summaries, "test_user": test_user})

    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/recommend", response_class=HTMLResponse)
def recommend(request: Request, user_id, n=9):
    try:
        predictions, titles, summaries, urls, categories, history, history_urls = [], [], [], [], [], [], []

        # Loading News Dataset
        path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data'), 'processed')
        news_df = pd.read_csv(path + '/Category_df.csv').fillna("")
        news_df = news_df.sort_values(by=['rating'], ascending=False)

        # Loading Summary Dataset
        path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data'), 'summary')
        summary_df = pd.read_pickle(path + '/LexRankSummary.p')
        summary_df = summary_df[['News ID', 'Cleaned_Article', 'Summary (TextRank)', 'Summary (LexRank)']]

        # Merging News With Summary Dataset
        news_df = news_df.merge(summary_df, how='left', left_on='News ID', right_on='News ID').fillna("")
        news_df = news_df.replace("Nan", "").replace("nan", "")
        news_df = news_df.sort_values(by=['rating'], ascending=False)

        # Load Users' Predictions
        predictions = trainer.get_top_n(user_id, int(n))

        if not predictions:
            # Use Cold Start Option
            user_id += " (New User)"
            top_news = news_df["News ID"][:n]

            for article in top_news:
                categories.append(str(news_df[news_df['News ID'] == article].iloc[0]['Category']).capitalize())
                titles.append(str(news_df[news_df['News ID'] == article].iloc[0]['Title']))
                urls.append(str(news_df[news_df['News ID'] == article].iloc[0]['URL']))

                # (Summary of TextRank) Then (Abstract Summary from dataset) Then (Article) Then (Empty String)
                if news_df[news_df['News ID'] == article].iloc[0]['Summary (TextRank)'] != "":
                    summaries.append(str(news_df[news_df['News ID'] == article].iloc[0]['Summary (TextRank)']).capitalize())
                elif news_df[news_df['News ID'] == article].iloc[0]['Abstract'] != "":
                    summaries.append("Abstract Summary :- " + str(news_df[news_df['News ID'] == article].iloc[0]['Abstract']).capitalize())
                elif news_df[news_df['News ID'] == article].iloc[0]['Cleaned_Article'] != "":
                    summaries.append("Article :- " + str(
                        news_df[news_df['News ID'] == article].iloc[0]['Cleaned_Article']).capitalize())
                else:
                    summaries.append("")

            return temps.TemplateResponse("home.html",
                                          {"request": request, "user_id": user_id,
                                           "predictions": top_news, "categories": categories,
                                           "titles": titles, "summaries": summaries, "urls": urls, "history": history,
                                           "history_urls": history_urls})
        else:
            # if loaded recommendation less than the required printing spaces
            if len(predictions['Articles']) < (n+1):
                extension = (n+1) - len(predictions['Articles'])
                top = news_df['News ID'][:extension].tolist()
                predictions['Articles'].extend(top)

            for article in predictions['Articles']:
                categories.append(str(news_df[news_df['News ID'] == article].iloc[0]['Category']).capitalize())
                titles.append(str(news_df[news_df['News ID'] == article].iloc[0]['Title']))
                urls.append(str(news_df[news_df['News ID'] == article].iloc[0]['URL']))

                # (Summary of TextRank) Then (Abstract Summary from dataset) Then (Article) Then (Empty String)
                if news_df[news_df['News ID'] == article].iloc[0]['Summary (TextRank)'] != "":
                    summaries.append(str(news_df[news_df['News ID'] == article].iloc[0]['Summary (TextRank)']).capitalize())
                elif news_df[news_df['News ID'] == article].iloc[0]['Abstract'] != "":
                    summaries.append("Abstract Summary :- " + str(news_df[news_df['News ID'] == article].iloc[0]['Abstract']).capitalize())
                elif news_df[news_df['News ID'] == article].iloc[0]['Cleaned_Article'] != "":
                    summaries.append("Article :- " + str(news_df[news_df['News ID'] == article].iloc[0]['Cleaned_Article']).capitalize())
                else:
                    summaries.append("")

            # Loading Users' History
            path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data'), 'processed')
            df = pd.read_csv(path + '/df_unpivoted.csv').fillna("")
            df = df[(df["user_id"] == user_id) & (df["rating"] > 0)]

            # Merging History with News Dataset
            merged_df = df.merge(news_df, how='left', left_on='item_id', right_on='News ID')
            history = (merged_df["Title"] + " (" + merged_df["Category"] + ")").tolist()
            history_urls = merged_df["URL"].tolist()

            return temps.TemplateResponse("home.html",
                                          {"request": request, "user_id": user_id, "history_urls": history_urls,
                                           "predictions": predictions['Articles'], "categories": categories,
                                           "titles": titles, "summaries": summaries, "urls": urls, "history": history})

    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/status", summary="Get model status")
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
