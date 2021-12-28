import pandas as pd
from rouge import Rouge
from nltk import download
from nltk.stem import WordNetLemmatizer
from Summarization.app import Summarizer
from Summarization.utils import Article
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from transformers import BartTokenizer, BartForConditionalGeneration

rouge = Rouge()
download('stopwords')
lem = WordNetLemmatizer()

# 1) Get the Data
newsDF = pd.read_pickle('./Data/processed/cleaned_news.p').reset_index().drop(columns='index')
newsDF = newsDF[['News ID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'Article', 'Cleaned_Article']]

# 2) Clean the Data
newsDF["Cleaned_Article"] = newsDF['Article'].apply(lambda x: Article.cleaningArticle("default", str(x)))

# Applying TextRank summarization on data
newsDF['Summary (TextRank)'] = newsDF["Cleaned_Article"].apply(lambda x: Summarizer.extracitve(x, TextRankSummarizer, 2))

# Applying LexRank on data
newsDF['Summary (LexRank)'] = newsDF["Cleaned_Article"].apply(lambda x: Summarizer.extracitve(x, LexRankSummarizer, 2))

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

newsDF['Summary (BART)'] = newsDF["Cleaned_Article"].apply(lambda x: Summarizer.abstractive(x, tokenizer, model))
newsDF.to_pickle('../Data/summary/LexRankSummary.p')
