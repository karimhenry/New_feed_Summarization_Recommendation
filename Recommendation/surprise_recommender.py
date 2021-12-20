import os
import json
import joblib
import random
import pandas as pd
from datetime import datetime
from collections import defaultdict
from surprise import Dataset, SVD, Reader, accuracy
from surprise.model_selection import train_test_split, GridSearchCV
from typing import Dict, List


class surprise_L:

    def dataloader(df, rating_lb=0, rating_ub=1):
        reader = Reader(rating_scale=(rating_lb, rating_ub))
        dataDF = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)

        # sample random train set and test set, test set is made of 20% of the ratings.
        trainset, testset = train_test_split(dataDF, test_size=.2)
        return trainset, testset

    def get_top_n(predictions, n=10):
        """Return the top-N recommendation for each auth from a set of predictions.

        Args:
            predictions(list of Prediction objects): The list of predictions, as
                returned by the test method of an algorithm.
            n(int): The number of recommendation to output for each auth. Default
                is 10.

        Returns:
        A dict where keys are auth (raw) ids and values are lists of tuples:
            [(raw item id, rating estimation), ...] of size n.
        """

        # First map the predictions to each auth.
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            # top_n[uid].append((iid, est))
            top_n[uid].append(iid)

        # Then sort the predictions for each auth and retrieve the k the highest ones.
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n]

        # top_n[uid] = set(top_n[uid])
        # Print the recommended items for each auth
        #     for uid, user_ratings in top_n.items():
        #         print(uid, [iid for (iid, _) in user_ratings])
        # top_n = {'Top 10 articles for you':set(top_n[uid])}

        return top_n

    def precision_recall_at_k(predictions, k=2, threshold=0.2):
        """Return precision and recall at k metrics for each auth"""

        # First map the predictions to each auth.
        user_est_true = defaultdict(list)
        for uid, _, true_r, est, _ in predictions:
            user_est_true[uid].append((est, true_r))

        precisions = dict()
        recalls = dict()
        for uid, user_ratings in user_est_true.items():
            # Sort auth ratings by estimated value
            user_ratings.sort(key=lambda x: x[0], reverse=True)

            # Number of relevant items
            n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

            # Number of recommended items in top k
            n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

            # Number of relevant and recommended items in top k
            n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold)) for (est, true_r) in user_ratings[:k])

            # Precision@K: Proportion of recommended items that are relevant
            # When n_rec_k is 0, Precision is undefined. We here set it to 0.
            precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

            # Recall@K: Proportion of relevant items that are recommended
            # When n_rel is 0, Recall is undefined. We here set it to 0.
            recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

        return precisions, recalls


class SurpriseTrainer:
    def __init__(self, mode=0) -> None:
        self.library = 'surprise'

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

        # self._running_threads = []
        # self._pipeline = None

    def get_status(self) -> Dict:
        return self.model_status

    # def _train_job(self, x_train: List[str], x_test: List[str], y_train: List[Union[str, int]],
    #                y_test: List[Union[str, int]]):
    #     self._pipeline.fit(x_train, y_train)
    #     report = classification_report(
    #         y_test, self._pipeline.predict(x_test), output_dict=True)
    #     classes = self._pipeline.classes_.tolist()
    #     self._update_status("Model Ready", classes, report)
    #     joblib.dump(self._pipeline, self.__model_path, compress=9)
    #     self.model = self._pipeline
    #     self._pipeline = None
    #     thread_id = get_ident()
    #     for i, t in enumerate(self._running_threads):
    #         if t.ident == thread_id:
    #             self._running_threads.pop(i)
    #             break

    def train(self) -> None:
        # if len(self._running_threads):
        #     raise Exception("A training process is already running.")

        # if self.library == 0:
        self.trainset, self.testset = surprise_L.dataloader(self.data)

        algo = SVD(n_factors=100, n_epochs=10, biased=True, init_mean=0, init_std_dev=0.1,
                   lr_all=0.005, reg_all=0.02, lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None,
                   reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None, random_state=42, verbose=True, )

        self._model = algo.fit(self.trainset)

        self.predictions1 = algo.test(self.testset)

        # if os.path.exists(self.__model_path):
        # self.model = joblib.load(self.__model_path)
        # self.predictions1 = self.model.test(self.testset)

        precisions1, recall1 = surprise_L.precision_recall_at_k(self.predictions1, k=20, threshold=0.15)
        self.pre = (sum(prec for prec in precisions1.values()) / len(precisions1))
        print(f'Training finished with precision {self.pre}')

        joblib.dump(self._model, self.__model_path, compress=9)

        self._update_status(self.library+' '+self.mode+" Model Ready",  self.pre)
        # self.model = self._pipeline

        # elif self.library == 1:
        #     mf = matrix_factorization('article')
        # update model status
        # self.model = None
        # self._update_status("Training")
        #
        # t = Thread(target=self._train_job, args=(trainset, testset))
        # self._running_threads.append(t)
        # t.start()

    def _predict(self) -> List[Dict]:
        # self.__predictions_path = os.path.join(self.__data_path, 'predictions')
        if not os.path.exists(self.__predictions_path):
            os.mkdir(self.__predictions_path)

        if os.path.exists(self.__model_path):
            self.model = joblib.load(self.__model_path)
        else:
            self.model = None

        if self.model:
            reader = Reader(rating_scale=(0, 1))
            dataDF = Dataset.load_from_df(self.data[['user_id', 'item_id', 'rating']].head(1000), reader)
            trainset = dataDF.build_full_trainset()

            print('shuffle in memo')
            testset = trainset.build_anti_testset()
            print('end of shuffle in memo')

            self.predictions1 = self.model.test(testset)
            top_preds = (surprise_L.get_top_n(self.predictions1, n=50))
            with open(os.path.join(self.__predictions_path, 'predictions_' + self.mode + '.json'), 'w') as f:
                json.dump(top_preds, f)
            return top_preds

        #   probs = self.model.predict_proba(texts)
        #     for i, row in enumerate(probs):
        #         row_pred = {}
        #         row_pred['text'] = texts[i]
        #         row_pred['predictions'] = {class_: round(float(prob), 3) for class_, prob in zip(
        #             self.model_status['classes'], row)}
        #         response.append(row_pred)
        # else:
        #     raise Exception("No Trained model was found.")
        # return response

    def _update_status(self, status: str, evaluation: int) -> None:
        self.model_status['status'] = status
        self.model_status['timestamp'] = datetime.now().isoformat()
        self.model_status['precision'] = evaluation

        with open(self.__status_path, 'w+') as file:
            json.dump(self.model_status, file, indent=2)

    def get_top_n(self, user_id, n):
        if user_id in self.preds.keys():
            return {f'Articles': self.preds[user_id][:n]}
        else:
            return False

    def gridsearch(self) -> None:
        # # if len(self._running_threads):
        # #     raise Exception("A training process is already running.")
        #
        # # if self.library == 0:
        # self.trainset, self.testset = surprise_L.dataloader(self.data)
        #
        # algo = SVD(n_factors=100, n_epochs=10, biased=True,init_mean=0, init_std_dev=0.1,
        #            lr_all=0.005, reg_all=0.02, lr_bu=None,lr_bi=None, lr_pu=None,lr_qi=None,
        #            reg_bu=None,reg_bi=None, reg_pu=None,reg_qi=None, random_state=42,verbose=True, )
        #
        # self._model = algo.fit(self.trainset)
        #
        # self.predictions1 = algo.test(self.testset)
        # # if os.path.exists(self.__model_path):
        # # self.model = joblib.load(self.__model_path)
        # # self.predictions1 = self.model.test(self.testset)
        # precisions1, recall1 = surprise_L.precision_recall_at_k(self.predictions1, k=20, threshold=0.15)
        # self.pre = (sum(prec for prec in precisions1.values()) / len(precisions1))
        # print(f'Training finished with precision {self.pre}')

        # Load the full dataset.
        reader = Reader(rating_scale=(0, 1))
        data = Dataset.load_from_df(self.data[['user_id', 'item_id', 'rating']], reader)

        raw_ratings = data.raw_ratings

        # shuffle ratings if you want
        random.shuffle(raw_ratings)

        # A = 90% of the data, B = 10% of the data
        threshold = int(.9 * len(raw_ratings))
        A_raw_ratings = raw_ratings[:threshold]
        B_raw_ratings = raw_ratings[threshold:]

        data.raw_ratings = A_raw_ratings  # data is now the set A

        # Select your best algo with grid search.
        print('Grid Search...')
        param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005]}
        grid_search = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
        grid_search.fit(data)

        algo = grid_search.best_estimator['rmse']

        # retrain on the whole set A
        trainset = data.build_full_trainset()
        algo.fit(trainset)

        # Compute biased accuracy on A
        predictions = algo.test(trainset.build_testset())
        print('Biased accuracy on A,', end='   ')
        accuracy.rmse(predictions)
        precisions1, recall1 = surprise_L.precision_recall_at_k(predictions, k=20, threshold=0.15)
        print('precision: ', sum(prec for prec in precisions1.values()) / len(precisions1))

        # Compute unbiased accuracy on B
        testset = data.construct_testset(B_raw_ratings)  # testset is now the set B
        predictions = algo.test(testset)
        print('Unbiased accuracy on B,', end=' ')
        accuracy.rmse(predictions)
        precisions1, recall1 = surprise_L.precision_recall_at_k(predictions, k=20, threshold=0.15)
        print('precision: ', sum(prec for prec in precisions1.values()) / len(precisions1))


# ====Code Running====
# a = SurpriseTrainer(0)
# a.train()
# print(a.get_status())
# predictions = a.get_top_n('U1000', 10)
# print(predictions)
