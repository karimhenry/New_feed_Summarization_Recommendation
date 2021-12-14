from sklearn.model_selection import train_test_split
from matrix_factorization import BaselineModel, KernelMF, train_update_test_split
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd

class matrix_factorization():

    def __init__(self,mode):
        self.mode=mode
        valid = {'article', 'subcat'}
        if self.mode not in valid:
            raise ValueError("results: Library must be one of %r." % valid)

    def Articles_Train_Test(self,df, User=None):
        # Train test split function
        if self.mode=='article':
            (self.X_train_initial, self.y_train_initial, self.X_train_update, self.y_train_update, self.X_test_update,
             self.y_test_update) = train_update_test_split(df, frac_new_users=0.2)

        elif self.mode =='subcat':
            (self.X_train_initial, self.y_train_initial, self.X_train_update, self.y_train_update, self.X_test_update,
             self.y_test_update) = self.train_update_test_split_modified(df, frac_new_users=0.2)

        # Initial training where our rating is between 0 and 1
        matrix_fact = KernelMF(n_epochs=20, min_rating=0, max_rating=1, verbose=0)
        matrix_fact.fit(self.X_train_initial, self.y_train_initial)

        # Update model with new users
        matrix_fact.update_users(self.X_train_update, self.y_train_update, verbose=0)
        pred = matrix_fact.predict(self.X_test_update)

        # Evalutate Model
        rmse = mean_squared_error(self.y_test_update, pred, squared=False)
        print(f"Articles Test RMSE: {rmse:.4f}")

        return (matrix_fact)

    def train_update_test_split_modified(self,X, frac_new_users) :
        """
        We Modified the library Train Test Split function to remove
        Startify as some records contained only 1 users at subcategory

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
        users_update = np.random.choice(users, size=round(frac_new_users*len(users)),replace=False)

        # Initial training matrix
        train_initial = X.query("user_id not in @users_update").sample(frac=1,replace=False)

        # Train and test sets for updating model. For each new auth split their ratings into two sets, one for update and one for test
        data_update = X.query("user_id in @users_update")
        train_update, test_update = train_test_split(data_update, test_size=0.5)

        # Split into X and y
        X_train_initial, y_train_initial = (train_initial[["user_id", "item_id"]],train_initial["rating"])
        X_train_update, y_train_update = (train_update[["user_id", "item_id"]],train_update["rating"])
        X_test_update, y_test_update = (test_update[["user_id", "item_id"]],test_update["rating"])

        return (X_train_initial,y_train_initial,X_train_update,y_train_update,X_test_update,y_test_update)
