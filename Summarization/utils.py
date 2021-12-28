import re
import nltk
import trafilatura
import contractions
import pandas as pd
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class Article:
    def scraper(url):
        """Extract text from URL """
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(downloaded)
        return text

    def loading(filepath):
        """ Loading Articles from folders in returned as Dataframe """

        filepath = Path(filepath)

        def extract(filepath):
            """ Extract text from files """
            pattern = r'(\w+)/(\d+)\.txt$'
            category, file_id = re.search(pattern, str(filepath)).groups()
            with open(filepath, 'r', encoding='unicode_escape') as f:
                text = f.read()
                return category, file_id, text

        articles_data = list(map(extract, filepath.glob('News Articles/*/*.txt')))
        summaries_data = list(map(extract, filepath.glob('Summaries/*/*.txt')))

        articles_df = pd.DataFrame(articles_data, columns=('Category', 'ID', 'Article'))
        articles_df['Article'] = articles_df['Article'].str.replace("\n\n", " , ")
        summaries_df = pd.DataFrame(summaries_data, columns=('Category', 'ID', 'Summary'))
        df = articles_df.merge(summaries_df, how='inner', on=('Category', 'ID'))

        return df

    def cleaningArticle(ArticleOrSummary, row):
        """Function to clean article and summary"""
        lem = WordNetLemmatizer()
        if ArticleOrSummary == "article":
            row = row.split(",")
            row = " ,".join(row[1:])

        row = row.lower()
        row = re.sub(r"""[<>_()|&=\‘”"“``\+\"©ø/\[\]\\;?~*!]""", " ", str(row))
        row = contractions.fix(row)  # handling the Contractions
        CURRENCIES = {"$": "USD", "zł": "PLN", "£": "GBP", "¥": "JPY", "฿": "THB", "₡": "CRC", "₦": "NGN", "₩": "KRW", "₪": "ILS", "₫": "VND", "€": "EUR", "₱": "PHP", "₲": "PYG", "₴": "UAH", "₹": "INR"}
        CURRENCY_REGEX = re.compile("({})+".format("|".join(re.escape(c) for c in CURRENCIES.keys())))
        EMAIL_REGEX = re.compile(r"(?:^|(?<=[^\w@.)]))([\w+-](\.(?!\.))?)*?[\w+-]@(?:\w-?)*?\w+(\.([a-z]{2,})){1,3}(?:$|(?=\b))", flags=re.IGNORECASE | re.UNICODE)

        row = EMAIL_REGEX.sub(' ', str(row))  # remove any email
        row = CURRENCY_REGEX.sub(' ', str(row))  # currency handling
        row = re.sub("(@[A-Za-z0-9]+)", ' ', str(row))  # remove any hashtags
        row = re.sub("(#[A-Za-z0-9]+)", ' ', str(row))  # remove any mentions
        row = re.sub(r'http\S+', '', str(row))  # remove any url

        # separate with no blunder MachineLeaning -> Machine Learning
        row = ' '.join(re.findall('[A-Za-z]*[^A-Z]*', str(row)))
        row = ' '.join(re.findall('[(A-Z)-a-z]*[^A-Z]*', str(row)))
        row = " ".join(nltk.word_tokenize(row))

        # row = re.sub(r'[.\s][[A-Za-z0-9]*\.(com|org|edu|gif|net)','', str(row)) # remove any website link
        row = re.sub(r'(--)+', ' ', str(row))  # remove any (--)or more come together
        row = re.sub(r'((P|p)([.])(M|m))|((A|a)([.])(M|m))', lambda x: x.group().replace(".", ""), str(row))  # P.M -> PM
        row = re.sub(r'\s:\s', ' ', str(row))

        # 3 cases to separate by (-) -> (word-digit)(word-word)(digit-word)
        row = re.sub(r'(\d*[-]\w*)|(\w*[-]\w*)|(\w*[-]\d*)', lambda x: x.group().replace("-", " "), str(row))
        row = re.sub(r'((\w+)([.])(\w+))', lambda x: x.group().replace(".", ""), str(row))  # take.note -> take note
        row = re.sub(r'([.])(\w+)', lambda x: x.group().replace(".", " . "), str(row))  # .first -> first
        row = re.sub(r'(\w+)([.])', lambda x: x.group().replace(".", " . "), str(row))
        row = nltk.sent_tokenize(row)
        row = " ".join([lem.lemmatize(word) for word in row if word not in stopwords.words('english')])
        row = re.sub(' +', ' ', str(row))  # remove multi spaces or tap
        row = re.sub(r"([\s])'s", "'s", row)

        return row.capitalize()
