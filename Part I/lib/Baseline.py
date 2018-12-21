"""
This file contaings the `Baseline` class implementing the baseline algorithm
that predicts ratings by averaging over item ratings and user ratings
"""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin

from lib import utils


class Baseline(BaseEstimator, RegressorMixin):  
     
    def __init__(self, unique_users = 0, 
                 unique_items = 0,
                 response_vector = None):
        """
        This method is called when initializing the estimator
        and assigns values to the object's parameters
        Args:
            - self: Baseline object, to be initialized
            - unique_users: int, the number of unique users
            - unique_items: int, the number of unique items
            - response_vector: array, used by the GridSearchCV class
            of scikit-learn, kept equal to None in our work
        """
        self.response_vector = response_vector
        self.unique_users = unique_users
        self.unique_items = unique_items
            
    def fit(self, X, y):
        """
        This methods fits the algorithm to the data
        Args:
            - self: Baseline object, to be fitted
            - X: panda DataFrame, with hashed available pairs of item-user
            - y: panda Series, with corresponding ratings to the
            pairs of item-user in X
        """
        
        #remove mean
        data = pd.DataFrame({'user' : X['user'],
                             'item' : X['item'],
                             'rating' : y - np.mean(y)})
        
        # Useful variables
        nb_users = self.unique_users
        nb_items = self.unique_items
        
        # Initialize biases
        b_items = np.zeros(nb_items)
        b_users = np.zeros(nb_users)
        
        # Visited users and items
        known_u = np.full(nb_users, False, dtype = bool)
        known_i = np.full(nb_items, False, dtype = bool)
        
        for user in range(1, nb_users + 1):
            
            ratings_for_user = data[data['user'] == user]['rating']
            if(ratings_for_user.size > 0):
                avg_user = np.average(data[data['user'] == user]['rating'])
                known_u[user - 1] = True
                b_users[user - 1] = avg_user
        
        for item in range(1, nb_items + 1):
            
            ratings_for_item = data[data['item'] == item]['rating']
            if(ratings_for_item.size > 0):
                avg_item = np.average(data[data['item'] == item]['rating'])
                known_i[item - 1] = True
                b_items[item - 1] = avg_item

        self.user_bias_ = b_users
        self.item_bias_ = b_items
        self.mean_ = np.mean(y)
        
        # store visited items and users
        self.known_users_ = known_u
        self.known_items_ = known_i

        return self
    
    def predict(self, X, y = None):
        """
        This function predicts the ratings for the data in X
        if the algorithm has been fitted, and throws an error otherwise
        Args:
            - self: MF object, should have called fit method earlier
            - X: panda DataFrame, with hashed pairs of item-user without 
            ratings
            - y: panda Series, needs to be supplied to insert in scikit-learn
            pipeline but kept to value `None`
        Output:
            - prediction: np array, containing the predicted ratings for
            each user-item pair in X
        """
        
        try:
            getattr(self, "user_bias_")
        except AttributeError:
            raise RuntimeError("You must fit the model before predicting.")
        
        # initialize predictions
        pred = np.zeros(X['user'].size)
        
        for index in range(X['user'].size):
            
            user = utils.get_user_at(index, X) - 1
            item = utils.get_item_at(index, X) - 1
            
            if(self.known_users_[user] == True and 
               self.known_items_[item] == True):
                biases = self.user_bias_[user] + self.item_bias_[item]
                pred[index] =  biases + self.mean_
            else:
                pred[index] = float('NaN')
        
        return pred
    
    def score(self, X, y):
        """
        This function returns the score obtained at the end of training
        Args:
            - self: MF object, should have called fit method earlier
            - X: panda DataFrame, with pairs of item-user with ratings
            - y: panda Series, contains the ratings associated with pairs in X
        Output:
            - score: float, the score on the training data (- RMSE)
        """
        size = utils.total_ratings(X)
        prediction = self.predict(X)
        err = 0
        real_size = 0
        for i in range(size):
            if(prediction[i] == prediction[i]): # check if NaN
                err += (prediction[i] - y[i])**2
                real_size += 1
        score = - np.sqrt(err / real_size)
        return score