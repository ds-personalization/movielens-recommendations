"""
This file contaings the `Baseline` class implementing the baseline algorithm
that predicts ratings by averaging over item ratings and user ratings
"""

import numpy as np
import pandas as pd

from lib import utils


class Baseline():  
     
    def __init__(self, unique_users = 0, 
                 unique_items = 0):
        """
        This method is called when initializing the estimator
        and assigns values to the object's parameters
        Args:
            - self: Baseline object, to be initialized
            - unique_users: int, the number of unique users
            - unique_items: int, the number of unique items
        """
        self.unique_users = unique_users
        self.unique_items = unique_items
            
    def fit(self, X, y):
        """
        This methods fits the algorithm to the data
        Args:
            - self: Baseline object, to be fitted
            - X: pandas DataFrame, with hashed available pairs of item-user
            - y: pandas Series, with corresponding ratings to the pairs of 
            item-user in X
        """
        
        # remove mean
        data = pd.DataFrame({'user' : X['user'],
                             'item' : X['item'],
                             'rating' : y - np.mean(y)})
        
        # useful variables
        nb_users = self.unique_users
        nb_items = self.unique_items
        
        # initialize biases
        b_items = np.zeros(nb_items)
        b_users = np.zeros(nb_users)
        
        # visited users and items
        known_u = np.full(nb_users, False, dtype = bool)
        known_i = np.full(nb_items, False, dtype = bool)
        
        # compute biases and mark visited users and items
        for user in range(nb_users):
            
            ratings_for_user = data[data['user'] == user]['rating']
            if(ratings_for_user.size > 0):
                avg_user = np.mean(data[data['user'] == user]['rating'])
                known_u[user] = True
                b_users[user] = avg_user
        
        for item in range(nb_items):
            
            ratings_for_item = data[data['item'] == item]['rating']
            if(ratings_for_item.size > 0):
                avg_item = np.mean(data[data['item'] == item]['rating'])
                known_i[item] = True
                b_items[item] = avg_item
                
        # store biases and mean
        self.user_bias_ = b_users
        self.item_bias_ = b_items
        self.mean_ = np.mean(y)
        
        # store visited items and users
        self.known_users_ = known_u
        self.known_items_ = known_i

        return self
    
    def predict(self, X):
        """
        This function predicts the ratings for the data in X
        if the algorithm has been fitted, and throws an error otherwise
        Args:
            - self: Baseline object, should have called fit method earlier
            - X: pandas DataFrame, with hashed pairs of item-user without 
            ratings
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
        
        # compute prediction for each user-item pair in X
        for index in range(X['user'].size):
            
            user = utils.get_user_at(index, X)
            item = utils.get_item_at(index, X)
            
            bias_i = 0
            bias_u = 0
            
            if(self.known_items_[item]):
                bias_i = self.item_bias_[item]
            if(self.known_users_[user]):
                bias_u = self.user_bias_[user]
                
            biases = bias_i + bias_u
            
            pred[index] =  biases + self.mean_

        return pred
    
    def score(self, X, y):
        """
        This function returns the score obtained at the end of training
        Args:
            - self: Baseline object, should have called fit method earlier
            - X: panda DataFrame, with pairs of item-user with ratings
            - y: panda Series, contains the ratings associated with pairs in X
        Output:
            - score: float, the score on the training data (- RMSE)
        """
        size = utils.total_ratings(X)
        prediction = self.predict(X)
        err = np.sum((y - prediction)**2)
        score = - np.sqrt(err / size)
        return score