import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from surprise import Dataset, Reader
from matrix_factorization import KernelMF, train_update_test_split


def train_update_test_split_modified(X, frac_new_users):
    """
    We Modified the library Train Test Split function to remove
    Stratify as some records contained only 1 users at subcategory

    Args:
        X (pd.DataFrame): Data frame containing columns user_id, item_id
        frac_new_users (float): Fraction of users to not include in train_initial

    Returns:
        X_train_initial [pd.DataFrame]: Training set user_ids and item_ids for initial model fitting
        y_train_initial [pd.Series]: Corresponding ratings for X_train_initial
        X_train_update [pd.DataFrame]: Training set user_ids and item_ids for model updating. Contains users that are not in train_initial
        y_train_initial [pd.Series]: Corresponding ratings for X_train_update
        X_test_update [pd.DataFrame]: Testing set user_ids and item_ids for model updating. Contains some users as train_update
        y_test_update [pd.Series]: Corresponding ratings for X_test_update
    """
    # Variable of unique users
    users = X["user_id"].unique()

    # Users that won't be included in the initial training
    users_update = np.random.choice(users, size=round(frac_new_users * len(users)), replace=False)

    # Initial training matrix
    train_initial = X.query("user_id not in @users_update").sample(frac=1, replace=False)

    # Train and test sets for updating model. For each new auth split their ratings into two sets, one for update and one for test
    data_update = X.query("user_id in @users_update")
    train_update, test_update = train_test_split(data_update, test_size=0.5)

    # Split into X and y
    X_train_initial, y_train_initial = (train_initial[["user_id", "item_id"]], train_initial["rating"])
    X_train_update, y_train_update = (train_update[["user_id", "item_id"]], train_update["rating"])
    X_test_update, y_test_update = (test_update[["user_id", "item_id"]], test_update["rating"])

    return X_train_initial, y_train_initial, X_train_update, y_train_update, X_test_update, y_test_update


class matrix_factorization:
    def __init__(self, mode=0) -> None:
        self.model = None
        self._model = None
        self.pre = None
        self.library = 'matrix_factorization'

        self.__data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data')

        # 0 for articles dataset and 1 for subcategories dataset
        valid = {0, 1}
        if mode not in valid:
            raise ValueError("results: Mode must be one of %r." % valid)

        self.__predictions_path = os.path.join(self.__data_path, 'predictions', 'prediction_' + self.library)
        self.__storage_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'storage',
                                           'storage_' + self.library)

        if not os.path.exists(self.__storage_path):
            os.mkdir(self.__storage_path)

        if mode == 0:
            data = 'df_unpivoted.csv'
            self.mode = 'articles'
            if os.path.exists(os.path.join(self.__predictions_path, 'predictions_articles.json')):
                with open(os.path.join(self.__predictions_path, 'predictions_articles.json')) as file:
                    self.preds = json.load(file)
            else:
                self.preds = None

            self.__status_path = os.path.join(self.__storage_path, 'model_status_articles.json')
            self.__model_path = os.path.join(self.__storage_path, 'model_pickle_articles.joblib')

        else:
            data = 'SubCat_DF.csv'
            self.mode = 'subcategories'

            if os.path.exists(os.path.join(self.__predictions_path, 'predictions_subcategories.json')):
                with open(os.path.join(self.__predictions_path, 'predictions_subcategories.json')) as file:
                    self.preds = json.load(file)
            else:
                self.preds = None

            self.__status_path = os.path.join(self.__storage_path, 'model_status_sub.json')
            self.__model_path = os.path.join(self.__storage_path, 'model_pickle_sub.joblib')

        self.processed_data_path = os.path.join(self.__data_path, 'processed', data)

        self.data = pd.read_csv(self.processed_data_path)

        if os.path.exists(self.__status_path):
            with open(self.__status_path) as file:
                self.model_status = json.load(file)
        else:
            self.model_status = {"status": "No Model found",
                                 "timestamp": datetime.now().isoformat(),
                                 "precision": 0}

    def _update_status(self, status: str, evaluation: int) -> None:
        self.model_status['status'] = status
        self.model_status['timestamp'] = datetime.now().isoformat()
        self.model_status['precision'] = evaluation

        with open(self.__status_path, 'w+') as file:
            json.dump(self.model_status, file, indent=2)

    def get_status(self) -> Dict:
        return self.model_status

    def train(self) -> None:
        # Train test split function
        if self.mode == 'articles':
            (X_train_initial, y_train_initial, X_train_update, y_train_update, X_test_update,
             y_test_update) = train_update_test_split(self.data, frac_new_users=0.2)
            
        elif self.mode == 'subcategories':
            (X_train_initial, y_train_initial, X_train_update, y_train_update, X_test_update,
             y_test_update) = train_update_test_split_modified(self.data, frac_new_users=0.2)

        else:
            return
            
        # Initial training where our rating is between 0 and 1
        matrix_fact = KernelMF(n_epochs=20, min_rating=0, max_rating=1, verbose=0)
        matrix_fact.fit(X_train_initial, y_train_initial)

        # Update model with new users
        matrix_fact.update_users(X_train_update, y_train_update, verbose=0)
        pred = matrix_fact.predict(X_test_update)

        # Evaluate Model
        rmse = mean_squared_error(y_test_update, pred, squared=False)
        print(f"Matrix Factorization RMSE: {rmse:.4f}")

        self._model = matrix_fact
        joblib.dump(self._model, self.__model_path, compress=9)
        self._update_status(self.library+' '+self.mode+" Model Ready", self.pre)

    def get_top_n(self, user_id, n):
        if not os.path.exists(self.__predictions_path):
            os.mkdir(self.__predictions_path)

        if os.path.exists(self.__model_path):
            self.model = joblib.load(self.__model_path)
            Recommend = self.model.recommend(user=user_id, amount=n)['item_id'].tolist()
            return {f'Articles': Recommend}
        else:
            return {f'Articles': ""}


# ====Code Running====
# a = matrix_factorization(0)
# a.train()
# print(a.get_status())
# predictions = a.get_top_n('U1000', 5)
# print(predictions['Articles'])

