# News Feed Summarization and Recommendation


Summarize key information of a given article and Recommend to user a set of summarized articles matching his/her interest

## About Dataset

The MIND dataset for news recommendation was collected from anonymized behavior logs of Microsoft News website. The data randomly sampled 1 million users who had at least 5 news clicks during 6 weeks from October 12 to November 22, 2019. To protect user privacy, each user is de-linked from the production system when securely hashed into an anonymized ID. Also collected the news click behaviors of these users in this period, which are formatted into impression logs. The impression logs have been used in the last week for test, and the logs in the fifth week for training. For samples in training set, used the click behaviors in the first four weeks to construct the news click history for user modeling. Among the training data, the samples in the last day of the fifth week used as validation set. This dataset is a small version of MIND (MIND-small), by randomly sampling 50,000 users and their behavior logs. Only training and validation sets are contained in the MIND-small dataset.

## Technologies Used

**Recommendations** Matrix Factorization

**Summarization** Extractive and Abstractive

[More info here](https://docs.google.com/presentation/d/1536WLXjunobkA0Jt83VhAaOTKIWGRsUg/edit?usp=sharing&ouid=114999608387157692062&rtpof=true&sd=true).

Project Organization
------------

    
    
    ├── README.md          <- The top-level README for developers using this project.
    ├── Data
    │   ├── prediction     <- Prediction for all users
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── Data_Preprocessing <- Scripts to process raw data  for modeling
    │   └── DataProcessing.py
    │
    ├── templates          <- contains .html files
    ├── Recommendation     <- Makes src a Python module
    │   ├── __init__.py
    │   ├── MF_R.py
    │   ├── Surprise_R.py
    │   ├── recommender.py
    │   └── Router.py
    │
    ├── storage     <- contains serialized model 
        ├── model_pickle.joblib
        └── model_status.json



--------
