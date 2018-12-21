"""
This file creates a class NDCG that implement functions to compute the NDCG.
"""

import numpy as np
import pandas as pd
import math
from tqdm import tqdm

class NDCG():
    def __init__(self, upper_bound = 10):
        """
        Method called when initilaizing the NDCG object.
        """
        self.upper_bound = upper_bound

    def create_ranking(self, X_train, X_test, y_train, y_test):
        """

        """
        train_true = X_train
        train_true['rating'] = y_train
        test_true = X_test
        test_true['rating'] = y_test

        table_true = pd.concat([train_true, test_true])

        #creating the dictionary of ranks for each user
        dico_ranking = {}
        unique_users = table_true.user.unique()
        for user in tqdm(unique_users):
            table_user = table_true[table_true['user'] == user]

            # retrieving ratings and items
            ratings = table_user.rating.values
            items = table_user.item.values

            # sorting them
            ratings, items = zip(*reversed(sorted(zip(ratings, items))))
            temp = {}
            for i in range(len(ratings)):
                temp[items[i]] = i + 1
            dico_ranking[user] = temp
        self.dico_ranking_ = dico_ranking

    def DCG(self, X_train, X_test, y_train, y_test):
        """
        This function computes the DCG for
        """
        try:
            getattr(self, "dico_ranking_")
        except AttributeError:
            raise RuntimeError("You must run create_ranking before computing DCG.")

        # creating the table
        train = X_train
        train['rating'] = y_train
        test = X_test
        test['rating'] = y_test

        table = pd.concat([train, test])

        DCG = 0
        unique_users = table.user.unique()
        for user in tqdm(unique_users):
            table_user = table[table['user'] == user]
            ratings = table_user.rating.values
            items = table_user.item.values
            ratings, items = zip(*reversed(sorted(zip(ratings, items))))
            temp = 0
            for i in range(min(len(ratings), self.upper_bound)):
                temp+= (2**ratings[i] - 1)/math.log(self.dico_ranking_[user][items[i]]+1,2)
            DCG+= temp
        DCG = DCG/len(unique_users)
        return DCG

    def NDCG(self, X_train, X_test, y_train, y_test, y_train, y_test_predict):
        """
        This method computes the NDCG.
        """
        try:
            getattr(self, "dico_ranking_")
        except AttributeError:
            raise RuntimeError("You must run create_ranking before computing NDCG.")

        DCG = self.DCG(X_train, X_test, y_train, y_test_predict)
        IDCG = self.DCG(X_train, X_test, y_train, y_test)
        NDCG = DCG / IDCG
        self.NDCG_ = NDCG
        return NDCG
