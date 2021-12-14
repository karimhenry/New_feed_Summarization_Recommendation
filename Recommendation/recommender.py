from collections import defaultdict
import pandas as pd
from surprise import Dataset
from surprise import SVD, Reader
from surprise.model_selection import KFold
from fastapi.templating import Jinja2Templates
from surprise.model_selection import train_test_split

import json
import os
import json
from datetime import datetime
from threading import Thread, current_thread, get_ident
from typing import Dict, List, Union

import joblib

from Recommendation.surprise_R import surprise_L
from Recommendation.MF_R import matrix_factorization



class Trainer():
    def __init__(self, mode = 0) -> None:

        # 0 for articles dataset and 1 for subcategories dataset
        valid = {0, 1}
        if mode not in valid:
            raise ValueError("results: Mode must be one of %r." % valid)
        self.mode = mode

        if mode==0:
            data='df_unpivoted.csv'

        else:
            data='SubCat_DF.csv'

        self.__data_path = os.path.join(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))), 'Data')

        self.processed_data_path = os.path.join(self.__data_path,'processed',data)

        self.data = pd.read_csv(self.processed_data_path)

        self.__storage_path = os.path.join(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))), 'storage')
        if not os.path.exists(self.__storage_path):
            os.mkdir(self.__storage_path)


        self.__status_path = os.path.join(
            self.__storage_path, 'model_status.json')
        self.__model_path = os.path.join(
            self.__storage_path, 'model_pickle.joblib')
        #
        if os.path.exists(self.__status_path):
            with open(self.__status_path) as file:
                self.model_status = json.load(file)
        else:
            self.model_status = {"status": "No Model found",
                                 "timestamp": datetime.now().isoformat(),
                                 "precision":0}
        #


        self.__predictions_path = os.path.join(self.__data_path,'prediction','predictions.json')
        if os.path.exists(self.__predictions_path):
            with open(self.__predictions_path) as file:
                self.preds = json.load(file)
        else : self.preds=None


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

    def train(self, library=0) -> None:
        # if len(self._running_threads):
        #     raise Exception("A training process is already running.")

        # 0:'surprise, 1:'MF'
        valid = {0,1}
        if library not in valid:
            raise ValueError("results: Library must be one of %r." % valid)
        self.library = library

        if self.library == 0:
            self.trainset, self.testset = surprise_L.dataloader(self.data)

            algo = SVD(n_factors=100, n_epochs=20, biased=True,init_mean=0, init_std_dev=0.1,
                       lr_all=0.005, reg_all=0.02, lr_bu=None,lr_bi=None, lr_pu=None,lr_qi=None,
                       reg_bu=None,reg_bi=None, reg_pu=None,reg_qi=None, random_state=42,verbose=True, )

            self._model = algo.fit(self.trainset)

            self.predictions1 = algo.test(self.testset)
            # if os.path.exists(self.__model_path):
            #     self.model = joblib.load(self.__model_path)
            self.predictions1 = self.model.test(self.testset)
            precisions1, recall1 = surprise_L.precision_recall_at_k(self.predictions1, k=20, threshold=0.15)
            self.pre = (sum(prec for prec in precisions1.values()) / len(precisions1))
            # print(f'Training finished with precision {pre}')
            # joblib.dump(self._model, self.__model_path, compress=9)
            self._update_status("Model Ready",  self.pre)

            # self.model = self._pipeline

        elif self.library == 1:
            mf = matrix_factorization('article')
        # update model status
        # self.model = None
        # self._update_status("Training")
        #
        # t = Thread(target=self._train_job, args=(
        #     trainset, testset))
        # self._running_threads.append(t)
        # t.start()

    def _predict(self) -> List[Dict]:

        self.__prediction_path = os.path.join(self.__data_path,'prediction')
        if not os.path.exists(self.__prediction_path):
            os.mkdir(self.__prediction_path )

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
            top_preds = ((surprise_L.get_top_n(self.predictions1, n=40)))
            with open(os.path.join(self.__prediction_path,'predictions.json'), 'w') as f:
                json.dump(top_preds, f)
            return top_preds
            # probs = self.model.predict_proba(texts)
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


    def get_top_n(self,user_id,n):

        if user_id in self.preds.keys():
            return {f'Top {n} articles for you ':self.preds[user_id][:n]}
        else:return {f'user not found'}

a = Trainer()
# print(a.model_status)
# a.train()
# print(a.pre)
# print(a._predict())
# print(a.get_status())

# print(a.get_top_10('U100',5))