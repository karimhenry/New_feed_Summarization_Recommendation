import os
import pandas as pd
from rouge import Rouge
from nltk import download
from nltk.stem import WordNetLemmatizer
from Summarization.app import Summarizer
from Summarization.utils import Article
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from transformers import BartTokenizer, BartForConditionalGeneration

download('stopwords')
rouge = Rouge()
lem = WordNetLemmatizer()

# Get the Data
newsDF = pd.read_csv(os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data'), 'raw') + '/news.tsv', sep='\t', header=None, names=['News ID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'Title Entities', 'Abstract Entities'])

# Scraping the Article using Trafilatura
newsDF['Article'] = newsDF['URL'].apply(lambda url: Article.scraper(url))

# Saving dataset after scraping
newsDF.to_pickle(os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data'), 'processed'), 'news_articles.p')

# Loading the Data after scraping articles
newsDF = pd.read_pickle(os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data'), 'processed'), 'news_articles.p').reset_index().drop(columns='index')

# Clean the Data
newsDF["Cleaned_Article"] = newsDF['Article'].apply(lambda x: Article.cleaningArticle("default", str(x)))

# Summarization :-
# -----------------

# Applying TextRank summarization on data
newsDF['Summary (TextRank)'] = newsDF["Cleaned_Article"].apply(lambda x: Summarizer.extracitve(x, TextRankSummarizer, 2))

# Applying LexRank summarization on data
newsDF['Summary (LexRank)'] = newsDF["Cleaned_Article"].apply(lambda x: Summarizer.extracitve(x, LexRankSummarizer, 2))

# Applying Bart summarization on data
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
newsDF['Summary (BART)'] = newsDF["Cleaned_Article"].apply(lambda x: Summarizer.abstractive(x, tokenizer, model))

# Saving the summaries results
newsDF.to_pickle(os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data'), 'summary'), 'LexRankSummary.p')
