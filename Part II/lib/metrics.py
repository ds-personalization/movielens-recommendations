"""
This file contains the custom evaluation metrics we use
to evaluate our recommendation systems
"""

import numpy as np
import pandas as pd
import math
import scipy.stats as sc

from tqdm import tqdm


def compute_top_k(model, X, k = 20, ratio= 0.001):
    """
    This function computes the top_k predictions using the model "model"
    for a ratio "ratio" of the rating matrix "X".
    Args:
        - model: Personalization model that implement the function predict.
        - X: pandas dataframe that has the columns : "old_user", "old_item",
         "user", "item", "rating".
        - k: int, number of top predictions to keep. Default value is 20.
        - ratio: float in [0,1], ratio of users for which to
        compute the top-k predictions. Default value is 0.001.
    Returns:
        - None
    """

    # find users
    users = X.user.unique()
    np.random.shuffle(users)
    dico_pred = {}

    # only computing the top-k for a fraction of the users
    # too expensive to run on the entire dataset
    for i in range(math.floor(len(users)*ratio)):
        user = users[i]
        u_df = predict_user(model, X, user)

        # delete the rows that are present in the train set
        # We do not want to recommend items that have already been consumed
        u_df = u_df[~u_df['old_item'].isin(X[X['user'] == user].old_item.values)]

        # keeping the top k predicted ratings
        u_df = u_df.nlargest(k, 'rating')

        # storing those predictions in the dictionary
        dico_pred[user] = u_df
    model.dict_top_k_= dico_pred

def predict_user(model, X, user):
    """
    This method predicts all item ratings for a specified user.
    Args:
        - model: recommender object that must have called the fit method.
        - X: panda dataframe, used to fit the model.
        - user: user id, user for which predictions will be made.
    Returns:
        - df: panda dataframe containing the predicted values for all items.
    """

    # retrieving all the items in the dataset
    unique_items = X.old_item.unique()

    # creating the dataframe to be passed to the predict method of model
    data = [[user, item] for item in unique_items]
    df = pd.DataFrame(data, columns = ['user', 'old_item'])

    # predicting all items for that user
    predictions = model.predict(df)
    df['rating'] = predictions
    return df

def compute_contingency_tables(model, X, epsilon = 0.001):
    """
    This function computes the contigency tables that will be used to make
    the Fisher test to evaluate the diversity of our model.
    Args:
        - model: Personalization model, should have computed the top_k
        - X: panda dataframe, ratings dataframe
        - epsilon: float, non-zero value to avoid dividing by zero when
        computing a ChiSquare test.
    """
    try:
        getattr(model, "dict_top_k_")
    except AttributeError:
        raise RuntimeError("You must have computed the top-k items first.")

    contigency_tables = {}

    # index makes the correspondance between cluster names and integer
    genres = [genre for genre in model.clusters.keys()]
    n_clusters = len(genres)

    for user in model.dict_top_k_.keys():
        # initialize a matrix full of epsilons, avoid dividing by 0 problems later
        matrix = np.full([2, n_clusters], epsilon)

        # We need to fill in the first column of our contigency table
        items = X[X['user'] == user].old_item.unique()
        for item in items:
            item_genres = model.membership[item]
            for genre in item_genres.keys():
                matrix[0][genres.index(genre)]+= 1

        # We need to fill in the second column of our contigency table
        pred_items = model.dict_top_k_[user].old_item.unique()
        for item in pred_items:
            item_genres = model.membership[item]
            for genre in item_genres.keys():
                matrix[1][genres.index(genre)]+= 1

        contigency_tables[user] = matrix
    model.contigency_tables_ = contigency_tables

def compute_serendipity(model):
    """
    This function computes a measure of serendipity that is based on
    ChiSquare tests on the contigency tables that have been created when calling the
    "compute_contigency_tables" function.
    Args:
        - model: MEMF object that must have called the
        compute_contigency_tables function.
    Output:
        - serendipity_score: a serendipity score that measure how novel are
        the model's predictions, regardless of their relevance.
    """
    try:
        getattr(model, 'contigency_tables_')
    except AssertionError:
        raise RuntimeError("You must call compute contigency tables before computing serendipity.")

    serendipity_score = 0
    for user in model.contigency_tables_.keys():
        table = model.contigency_tables_[user]
        chisq, _ = sc.chisquare(table[1], table[0])
        serendipity_score+= chisq
    model.serendipity_score_ = serendipity_score
    return serendipity_score

def compute_diversity(model, X, k = 20, epsilon = 0.001, ratio = 0.001):
    """

    """
    compute_top_k(model, X, k , ratio)
    compute_contingency_tables(model, X, epsilon)
    compute_serendipity(model)


class NDCG():
    
    
    def __init__(self, upper_bound = 10):
        """
        Method called when initilaizing the NDCG object.
        Args:
            - self: NDCG object, to be initialized
            - upper_bound: int, number of top rated items to consider.
                Default value is 10
        """
        self.upper_bound = upper_bound

    def create_ranking(self, X_train, X_test, y_train, y_test):
        """
        This function creates the item ranking for all users
        given their ratings on both the train and the test set.
        Args:
            - self: NDCG object
            - X_train: pandas dataframe, train dataset
            - X_test: pandas dataframe, test dataset
            - y_train: pandas Series, train values for ratings
            - y_test: pandas Series, test values for ratings
        Returns:
            - None
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

    def DCG(self, X_train, X_test, y_train, y):
        """
        This function computes the DCG for
        Args:
            - self: NDCG object, should have called create_ranking earlier.
            - X_train: pandas dataframe, train dataset
            - X_test: pandas dataframe, test dataset
            - y_train: pandas Series, train values for ratings
            - y_t pandas Series, test values for ratings
        Returns:
            - DCG: float, Discounted Cumulative Gain
        """
        try:
            getattr(self, "dico_ranking_")
        except AttributeError:
            raise RuntimeError("You must run create_ranking before computing DCG.")

        # creating the table
        train = X_train
        train['rating'] = y_train
        test = X_test
        test['rating'] = y

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

    def NDCG(self, X_train, X_test, y_train, y_test, y_test_predict):
        """
        This method computes the NDCG.
        Args:
            - self: NDCG object
            - X_train: pandas dataframe, train dataset
            - X_test: pandas dataframe, test dataset
            - y_train: pandas Series, train values for ratings
            - y_test: pandas Series, test values for ratings
            - y_test_predict: pandas Series, predicted values for ratings
        Returns:
            - NDCG: float in [0,1], Normalized Discounted Cumulative Gain
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

def coverage(model, X_test, nb_users, nb_items, 
             hashing_table = None, hs = False):
    """
    This function returns the coverage of items recommended
    based on predictions of `model` on `X_test` compared to
    `y_test`
    Args:
        - model: object that inherits the BaseEstimator
        class of scikit-learn, and that has already been
        fitted on some training data
        - X_test: pd DataFrame, the testing data
        - y_test: np array, the ratings corresponding
        to pairs in X_test
        - nb_users: int, the number of unique users
        - nb_items: int, the number of unique items
    Output:
        - coverage: float, the proportion of items recommended to
        at least one user out of all items
    """
    recommended_items = np.full(nb_items, False, dtype = bool)
    for user in range(nb_users):
        user_data = X_test[X_test['user'] == user]
        if(user_data['user'].size > 0):
            if not(hs):
                user_prediction = model.predict(user_data)
            else:
                user_prediction = model.predict(user_data, hashing_table)
            user_prediction_series = pd.Series(user_prediction)
            ranked_prediction = user_prediction_series.rank().astype(int)
            ranked_prediction.index = user_data.index
            user_data['ratings_rank'] = ranked_prediction
            reco_items_for_user = user_data.nlargest(10,'ratings_rank')['item']
            for item in reco_items_for_user:
                recommended_items[item] = True
    nb_recommended_items = np.sum(recommended_items)
    coverage = nb_recommended_items / nb_items
    return coverage