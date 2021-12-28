########################################################################################################################
from Data_Preprocessing.DataProcessing import DataPreprocess
from Recommendation.surprise_recommender import SurpriseTrainer
from Recommendation.matrix_factorization_recommender import matrix_factorization
########################################################################################################################
# Data PreProcessing :-
# ----------------------
Data = DataPreprocess()
Data.Preprocessing_Behaviors()  # Creates "df_unpivoted.csv" in "../Data/processed/"
Data.Preprocessing_News()  # Creates "SubCat_Df.csv" in "../Data/processed/"
Data.Preprocessing_Categories()  # Creates "Category_df.csv" in "../Data/processed/"

########################################################################################################################
# Article Matrix Factorization Recommender :-
# --------------------------------------------
MF_Recommender = matrix_factorization(0)  # 0 for Articles
MF_Recommender.train()  # Creates "model_pickle_articles.joblib" in "../storage/storage_matrix_factorization/"
print(MF_Recommender.get_status())  # Creates "model_pickle_articles.json" in "../storage/storage_matrix_factorization/"

########################################################################################################################
# SubCategory Matrix Factorization Recommender :-
# ------------------------------------------------
MF_Recommender = matrix_factorization(1)  # 1 for Subcategories
MF_Recommender.train()  # Creates "model_pickle_sub.joblib" in "../storage/storage_matrix_factorization/"
print(MF_Recommender.get_status())  # Creates "model_pickle_sub.json" in "../storage/storage_matrix_factorization/"

########################################################################################################################
# Article Surprise Recommender :-
# -------------------------------
Surprise_Recommender = SurpriseTrainer(0)  # 0 for Articles
Surprise_Recommender.train()  # Creates "model_pickle_articles.joblib" in "../storage/storage_surprise/"
Surprise_Recommender.gridsearch()  # Creates "model_pickle_articles.joblib" in "../storage/storage_surprise/"
Surprise_Recommender._predict()   # Creates "predictions_articles.json" in "../Data/predictions/predictions_surprise/"
print(Surprise_Recommender.get_status())  # Creates "model_pickle_articles.json" in "../storage/storage_surprise/"

########################################################################################################################
# SubCategory Surprise Recommender :-
# -----------------------------------
Surprise_Recommender = SurpriseTrainer(1)  # 1 for Subcategories
Surprise_Recommender.train()  # Creates "model_pickle_sub.joblib" in "../storage/storage_surprise/"
Surprise_Recommender.gridsearch()  # Creates "model_pickle_sub.joblib" in "../storage/storage_surprise/"
Surprise_Recommender._predict()   # Creates "predictions_sub.json" in "../Data/predictions/predictions_surprise/"
print(Surprise_Recommender.get_status())  # Creates "model_pickle_sub.json" in "../storage/storage_surprise/"
