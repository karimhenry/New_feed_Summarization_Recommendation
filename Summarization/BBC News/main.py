from fastapi import FastAPI, Request,Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd

newsDF=pd.read_pickle("./dataset/LexRankSummary.p")

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


templates = Jinja2Templates(directory="templates/")

@app.get("/")
def form_post(request: Request):
    
    return templates.TemplateResponse("home.html", {"request": request,})

@app.post("/")
def form_post(request: Request, newsID: str = Form(...)):
    if newsID in newsDF["News ID"].values:
        Title=str(newsDF[newsDF["News ID"]==newsID]['Title'].values[0])
        Category=str(newsDF[newsDF["News ID"]==newsID]['Category'].values[0])
        Summary=str(newsDF[newsDF["News ID"]==newsID]['Summary (LexRank)'].values[0]).capitalize()

        Article=str(newsDF[newsDF["News ID"]==newsID]['Cleaned_Article'].values[0])
        return templates.TemplateResponse("index.html", {"request": request,"Title":Title,
        "Category":Category,'Summary':Summary,"Article":Article ,'Article_ID':newsID})
    else:
        return templates.TemplateResponse("notFound.html", {"request": request})



# @app.get('/home',response_class=HTMLResponse)
# async def hello_world(request: Request):
#     return templates.TemplateResponse("home.html", {"request": request})

# @app.get('/index',response_class=HTMLResponse)
# async def get_summary(request: Request, newsID: str = Form(...)):
    
#     Title=str(newsDF[newsDF["News ID"]==newsID]['Title'].values[0])
#     Category=str(newsDF[newsDF["News ID"]==newsID]['Category'].values[0])
#     # Article=str(newsDF[newsDF["News ID"]==newsID]['Cleaned_Article'].values)
#     return templates.TemplateResponse("index.html", {"request": request,"Title":Title,
#     "Category":Category,"Article":"Mostafa" })
