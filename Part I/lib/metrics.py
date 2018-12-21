"""
This file contains the custom evaluation metrics we use
to evaluate our recommendation systems
"""

import numpy as np
import pandas as pd
import scipy.stats as sc
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split as t_t_split

def kendall_rank_coeff(model, X_test, y_test, nb_users):
    """
    This function returns the Kendall rank correlation
    coefficient based on predictions of `model` on
    `X_test` compared to `y_test`
    Args:
        - model: object that inherits the BaseEstimator
        class of scikit-learn, and that has already been
        fitted on some training data
        - X_test: pd DataFrame, the testing data
        - y_test: np array, the ratings corresponding
        to pairs in X_test
        - nb_users: int, the number of unique users
    Output:
        - avg_coeff: float, between -1 and 1, the kendall
        rank correlation coefficient averaged over all 
        users
    """
    coefficients = []
    data = pd.DataFrame({'user' : X_test['user'],
                         'item' : X_test['item'],
                         'rating' : y_test})
    for user in range(nb_users):
        data_for_user = data[data['user'] == user]
        if(data_for_user['user'].size > 1):
            
            user_prediction = model.predict(data_for_user)
            
            user_prediction_series = pd.Series(user_prediction)
            temp = user_prediction_series.rank(method = 'dense')
            ranked_prediction = temp.astype(int)
            
            temp = data_for_user['rating'].rank(method = 'dense')
            ranked_ratings = temp.astype(int)
            
            if(ranked_ratings.unique().size > 1):
                kendall_coeff = sc.kendalltau(ranked_prediction,
                                              ranked_ratings)
                coefficients.append(kendall_coeff.correlation)
            else:
                coefficients.append(0)
            
    avg_coeff = np.mean(coefficients)
    return avg_coeff

def kendall_rank_coeff_vectors(X_test, y_predict, y_test, nb_users):
    """
    This function returns the Kendall rank correlation
    coefficient between 'y_predict' and 'y_test'
    Args:
        - X_test: pd DataFrame, the testing data
        - y_predict: np array, the predicted ratings
        - y_test: np array, the true ratings
        - nb_users: int, the number of unique users
    Output:
        - avg_coeff: float, between -1 and 1, the kendall
        rank correlation coefficient averaged over all 
        users
    """

    coefficients = []

    data = pd.DataFrame({'user' : X_test['user'],
                         'item' : X_test['item'],
                         'rating' : y_test,
                         'pred_rating' : y_predict})
    
    for user in range(nb_users):
        
        data_for_user = data[data['user'] == user]
        
        if(data_for_user['user'].size > 1):
            
            temp = data_for_user['pred_rating'].rank(method = 'dense')
            ranked_prediction = temp.astype(int)
            
            temp = data_for_user['rating'].rank(method = 'dense')
            ranked_ratings = temp.astype(int)
            
            na_drop = pd.DataFrame({'pred' : ranked_prediction,
                                    'real' : ranked_ratings})
            na_drop.dropna()
            
            if(na_drop['real'].unique().size > 1 and
               na_drop['pred'].unique().size > 1):
                kendall_coeff = sc.kendalltau(na_drop['pred'],
                                              na_drop['real'])
                coefficients.append(kendall_coeff.correlation)
            
    avg_coeff = np.mean(coefficients)
    return avg_coeff

def coverage(model, X_test, nb_users, nb_items):
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
            user_prediction = model.predict(user_data)
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

def coverage_surprise(model, X_test, y_test, nb_users, nb_items):
    """
    This function returns the coverage of items recommended
    based on the list of prediction 'y_pred'
    Args:
        - model: surprise model
        - X_test: pd DataFrame, the testing data
        - y_test: pd Series, the ratings associated
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
            user_d = pd.DataFrame({'user' : user_data['user'], 
                                   'item' : user_data['item'], 
                                   'rating' : y_test})
            reader = Reader(rating_scale=(1, 5))
            user_d = Dataset.load_from_df(user_d, reader)
            _, user_d= t_t_split(user_d, test_size = 1.)
            user_prediction = model.test(user_d)
            clean_prediction = []
            for pred in user_prediction:
                clean_prediction.append(pred.est)
            user_prediction_series = pd.Series(clean_prediction)
            ranked_prediction = user_prediction_series.rank().astype(int)
            user_data['ratings_rank'] = ranked_prediction
            reco_items_for_user = user_data.nlargest(10,'ratings_rank')['item']
            for item in reco_items_for_user:
                recommended_items[item] = True
    nb_recommended_items = np.sum(recommended_items)
    coverage = nb_recommended_items / nb_items
    return coverage