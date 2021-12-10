from collections import defaultdict

from surprise.model_selection import train_test_split
from surprise import Dataset
from surprise import SVD, Reader
from surprise.model_selection import KFold
import numpy as np
import pandas as pd

class surprise_L():

    def dataloader(df, rating_lb=0, rating_ub=1):
        reader = Reader(rating_scale=(rating_lb, rating_ub))
        # data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)
        dataDF = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)

        # sample random trainset and testset
        # test set is made of 25% of the ratings.
        trainset, testset = train_test_split(dataDF, test_size=.25)
        return trainset, testset

    def get_top_n(predictions, n=10):
        """Return the top-N recommendation for each user from a set of predictions.

        Args:
            predictions(list of Prediction objects): The list of predictions, as
                returned by the test method of an algorithm.
            n(int): The number of recommendation to output for each user. Default
                is 10.

        Returns:
        A dict where keys are user (raw) ids and values are lists of tuples:
            [(raw item id, rating estimation), ...] of size n.
        """

        # First map the predictions to each user.
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            top_n[uid].append((iid, est))

        # Then sort the predictions for each user and retrieve the k highest ones.
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n]

        # Print the recommended items for each user
        #     for uid, user_ratings in top_n.items():
        #         print(uid, [iid for (iid, _) in user_ratings])

        return top_n

    def precision_recall_at_k(predictions, k=2, threshold=0.2):
        """Return precision and recall at k metrics for each user"""

        # First map the predictions to each user.
        user_est_true = defaultdict(list)
        for uid, _, true_r, est, _ in predictions:
            user_est_true[uid].append((est, true_r))

        precisions = dict()
        recalls = dict()
        for uid, user_ratings in user_est_true.items():
            # Sort user ratings by estimated value
            user_ratings.sort(key=lambda x: x[0], reverse=True)

            # Number of relevant items
            n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

            # Number of recommended items in top k
            n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

            # Number of relevant and recommended items in top k
            n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                                  for (est, true_r) in user_ratings[:k])

            # Precision@K: Proportion of recommended items that are relevant
            # When n_rec_k is 0, Precision is undefined. We here set it to 0.

            precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

            # Recall@K: Proportion of relevant items that are recommended
            # When n_rel is 0, Recall is undefined. We here set it to 0.

            recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0
        return precisions, recalls
