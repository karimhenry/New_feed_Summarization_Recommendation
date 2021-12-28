# News Feed Summarization and Recommendation
- Summarize key information of a given News feed and Recommend to user a set of summarized articles matching his/her interest

## About Dataset :-
- The MIND dataset for news recommendation was collected from anonymize behavior logs of Microsoft News website. The data randomly sampled 1 million users who had at least 5 news clicks during 6 weeks from October 12 to November 22, 2019. To protect user privacy, each user is de-linked from the production system when securely hashed into an anonymized ID. Also collected the news click behaviors of these users in this period, which are formatted into impression logs. The impression logs have been used in the last week for test, and the logs in the fifth week for training. For samples in training set, used the click behaviors in the first four weeks to construct the news click history for user modeling. Among the training data, the samples in the last day of the fifth week used as validation set. This dataset is a small version of MIND (MIND-small), by randomly sampling 50,000 users and their behavior logs. Only training and validation sets are contained in the MIND-small dataset.

## Technologies Used :-
- **Recommendations :-** Matrix Factorization and Surprise Libraries
- **Summarization :-** Extractive and Abstractive

###[Presentation Link](https://docs.google.com/presentation/d/1536WLXjunobkA0Jt83VhAaOTKIWGRsUg/edit?usp=sharing&ouid=114999608387157692062&rtpof=true&sd=true)

## Instructions :-
1) Install "requirement.txt" Packages :- 
   - "pip install -r requirements.txt"
2) Download "News Dataset", "Behaviour Dataset" and "LexRankSummary Dataset" Then Extract
   - [News Dataset](https://storage.googleapis.com/kaggle-data-sets/1049650/1765896/compressed/news.tsv.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20211223%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20211223T042710Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=3828f7a889b54b86598e056da14da70fdbc8d5b5a25eead3e7aa1c7f8873f886951f18f4b73b3914daef304be3d6283ab119eedc54fc7d5a8bfe8dfd12ff4cac5013012133376a98340f871804031502097151a176171b77bc5e7a4cc728a2ec9f6ad8710f4852b0b7417396177e3d25c45fe118d070070a1218a3c2ad1cbf69d3009d9958b27f0c4eb494558e6810d8fb956ed5e9793e83631f1367728932d220bbd232c0ccecacfc21fc7a49a230499aa41fe45256b8bc02a662ac6348e15bb68a57b1e26539e861b5a0b1c4cbf453f475d2031c7369b3fb79ed8c3324d5244a8095b5df40d8ea0da3415496f15925131e353b78abbc9cf082ccb4c6a31d65)
   - [Behaviour Dataset](https://storage.googleapis.com/kaggle-data-sets/1049650/1765896/compressed/behaviors.tsv.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20211223%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20211223T042820Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=0a3103fe6d4c7bb8d4175b44fd33e11e4ff3345faa0e7045a6e9c52079476b0dc9c199bdae190dfec10b47e941d2048d543e9249a54e20b0708ac136c399f17a82148c2281c7d1d39eb0ca6d6ae7d163c28b91b0cc05eafd24c866b1201c676734d33681ad75c8dab056975a37e3ff5a19828b5dd609600863293e0b38ad599646211cb993e6bb81106997d5c64c109634fce9e997f0f42542c23cf9561854d23c5198c758d458eb6eb10df0e2d4a267145bef33cb2ca768b35a52f55ad3fddaf7039ff081ec82a6f9e9e446f7044f5925518ee99ae959079c17a5034bbc8e64d239747547dd05211d9c5d77b8e475cf0669d4513d898a992df16c4ed7a23ba0)
   - [LexRankSummary Dataset](https://www.kaggle.com/melbahlun23/mindsummary?select=LexRankSummary_V1.p&fbclid=IwAR1WCgw1I-d6Id3ClyOWT9XHpXXN5fAr0QwqiEgmkRJJDT0dy22IBm5R71w)
3) Place "News.tsv" and "Behaviour.tsv" in "../Data/raw/"
4) Rename "LexRankSummary Dataset" to "LexRankSummary.p" and Place it in "../Data/Summary/"
5) Run "Recommender_Trainer.py" in "../Recommendation/" to train Recommender Models
6) Run "Main.py"

Project Organization :-
-----------------------
    ├── README.md          <- The top-level README for developers using this project.
    ├── Data
    │   ├── prediction     <- Prediction for all users
    │   │    ├── prediction_surprise
    │   │    │   ├── predictions_articles.json
    │   │    │   └── predictions_subcategories.json
    │   │    │
    │   │    └── prediction_matrix_factorization
    │   │
    │   ├── processed      <- Dataset after preprocessing.
    │   │    ├── Category_df.csv
    │   │    ├── df_unpivoted.csv
    │   │    └── SubCat_Df.csv
    │   │
    │   ├── raw            <- The original MIND dataset.
    │   │    ├── behaviours.tsv
    │   │    └── news.tsv
    │   │
    │   └── summary         <- Generated Summary using Lex Rank
    │        └── LexRankSummary.p
    │
    ├── Data_Preprocessing  <- Scripts to Preprocessing raw data for modeling
    │   └── DataProcessing.py
    │
    ├── Recommendation     <- Scripts to generate the model and predicted json arrays
    │   ├── matrix_factorization_recommender.py
    │   ├── surprise_recommender.py
    │   ├── Router.py
    │   └── Recommender_Trainer.py
    │
    ├── static     <- Template Assets,CSS and JS
    │   ├── assets
    │   │   └── favicon.ico
    │   ├── css
    │   │   └── styles.css
    │   └── js
    │       └── scripts.js
    │
    ├── storage     <- contains Generated model 
    │   ├── storage_surprise
    │   │   ├── model_pickle_sub.joblib
    │   │   ├── model_pickle_aricles.joblib
    │   │   ├── model_status_sub.json
    │   │   └── model_status_articles.json
    │   │
    │   └── matrix_fact
    │       ├── model_pickle_sub.joblib
    │       ├── model_pickle_aricles.joblib
    │       ├── model_status_sub.json
    │       └── model_status_articles.json
    │
    ├── templates          <- Contains .html files
    │   ├── home.html
    │   └── index.html
    │
    ├── main.py            <- Contains Main run file
    │
    ├── requirements.txt
    │
    └── run.sh

--------
