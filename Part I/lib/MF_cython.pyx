"""
This file contains the Cython implementation of the Matrix Factorization
with L2 regularization
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from six.moves import range

cimport cython

import numpy as np
cimport numpy as np 

from sklearn.base import BaseEstimator, RegressorMixin

import utils



class MF_cython(BaseEstimator, RegressorMixin):  
     
    def __init__(self, n_epochs = 20, n_factors = 20, 
                 learning_rate = .007, learning_rate_bias = .007,
                 lambd = 0.02,
                 unique_users = 0, unique_items = 0,
                 verbose = False,
                 random_state = None,
                 response_vector = None):
        """
        This method is called when initializing the estimator
        and assigns values to the object's parameters
        The default values where chosen to be the same
        as the SVD in the scikit `surprise` package 
        Args:
            - self: MF object, to be initialized
            - n_epochs: int, the number of epochs
            - n_factors: int, the number of factors in the
            latent user and item matrices
            - learning_rate: float, the learning rate for the
            SGD procedure used to find the user and item matrices
            - learning_rate_bias: float, the learning rate for the
            SGD procedure for the user and item biases
            - lambd: float, the L2 regularization parameter
            chosen to be the same for user and item vectors
            - unique_users: int, the number of unique users
            - unique_items: int, the number of unique items
            - verbose: boolean, if True prints the progress of 
            the fit procedure for training purposes
            - response_vector: array, used by the GridSearchCV class
            of scikit-learn, kept equal to None in our work
        """
        self.n_epochs = n_epochs
        self.n_factors = n_factors
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.learning_rate_bias = learning_rate_bias
        self.lambd = lambd
        self.response_vector = response_vector
        self.unique_users = unique_users
        self.unique_items = unique_items
    
    @cython.boundscheck(False) # turn off bounds-checking for function
    @cython.wraparound(False) # turn off negative index wrapping for function
    def fit(self, X, y):
        """
        This methods fits the algorithm to the data
        Args:
            - self: MF object, to be fitted
            - X: panda DataFrame, with hashed available pairs of item-user
            in columns `user` and `item`
            - y: panda Series, with corresponding ratings to the
            pairs of item-user in X
        """
        
        # Declarations
        
        # User and Item latent factors matrices
        cdef np.ndarray[np.double_t, ndim = 2] U
        cdef np.ndarray[np.double_t, ndim = 2] V
        
        # Bias vectors
        cdef np.ndarray[np.double_t] Bu
        cdef np.ndarray[np.double_t] Bi
        
        # Error vector
        cdef np.ndarray[np.double_t] err
        
        # Numpyp array versions of columns in X
        cdef np.ndarray[np.int16_t] user_column
        cdef np.ndarray[np.int16_t] item_column
        
        # Numpy array that will control the order of the updates
        cdef np.ndarray[np.long_t] random_indices
        
        # Visited users and items
        cdef np.ndarray[np.int8_t] known_u
        cdef np.ndarray[np.int8_t] known_i
        
        cdef int f, index
        cdef double dot, Uf, Vf
        cdef int nb_users, nb_items, size
        cdef double global_mean
        
        # aliases
        cdef double lr = self.learning_rate
        cdef double lrb = self.learning_rate_bias
        cdef double l = self.lambd

        # Initializations
        nb_users = self.unique_users
        nb_items = self.unique_items
        size = utils.total_ratings(X)
        
        Bu = np.zeros(nb_users, np.double)
        Bi = np.zeros(nb_items, np.double)
        
        U = np.random.rand(nb_users, self.n_factors)
        V = np.random.rand(nb_users, self.n_factors)
        
        user_column = np.array(X['user'], np.int16)
        item_column = np.array(X['item'], np.int16)
        
        err = np.zeros(size, np.double)
        
        known_u = np.zeros(nb_users, np.int8)
        known_i = np.zeros(nb_items, np.int8)
        
        global_mean = np.mean(y)

        # Run SGD to update the factors
        for epoch in range(self.n_epochs):
            
            if self.verbose and (epoch == 0 or (epoch + 1) % 10 == 0):
                print("Starting epoch {}".format(epoch + 1))
            
            random_indices = np.random.permutation(size)
            
            for index in random_indices:
                
                # -1 because hashing created users and items with integer
                # values starting at 1
                user = user_column[index] - 1
                item = item_column[index] - 1
                
                # mark user and item as visited
                known_u[user] = 1
                known_i[item] = 1
                
                dot = 0
                for f in range(self.n_factors):
                    dot += V[item, f] * U[user, f] 
                    
                # update error
                err[index] = y[index] - (global_mean + Bu[user] + Bi[item] + dot)
                
                # update biases
                Bu[user] += lrb * err[index]
                Bi[item] += lrb * err[index]
                
                # simultaneous update
                for f in range(self.n_factors):
                    Uf = U[user, f]
                    Vf = V[item, f]
                    U[user, f] += lr * (err[index] * Vf - l * Uf)
                    V[item, f] += lr * (err[index] * Uf - l * Vf)
           
        # store the results
        self.bias_user_ = Bu
        self.bias_item_ = Bi
        self.latent_users_ = U
        self.latent_items_ = V
        
        # store training mean
        self.global_mean_ = global_mean
        
        # store visited items and users
        self.known_users_ = known_u
        self.known_items_ = known_i
        
        # compute the score: -RMSE
        self.score_ = - np.sqrt(np.sum(err**2) / size)
        
        return self
    
    
    def predict(self, X, y = None):
        """
        This function predicts the ratings for the data in X
        if the algorithm has been fitted, and throws an error otherwise
        Args:
            - self: MF object, should have called fit method earlier
            - X: panda DataFrame, with pairs of item-user without ratings
            - y: panda Series, needs to be supplied to insert in scikit-learn
            pipeline but kept to value `None`
        Output:
            - prediction: np array, containing the predicted ratings for
            each user-item pair in X
        """
        
        try:
            getattr(self, "latent_users_")
        except AttributeError:
            raise RuntimeError("You must fit the model before predicting.")
        
        # initialize predictions
        prediction = np.zeros(X['user'].size)
        
        for index in range(X['user'].size):

            user = utils.get_user_at(index, X) - 1
            item = utils.get_item_at(index, X) - 1

            if(self.known_users_[user] == 1 and self.known_items_[item] == 1):
                dot = np.dot(self.latent_items_[item,], 
                             self.latent_users_[user,]) 
                bias = self.bias_user_[user] + self.bias_item_[item]
                prediction[index] = dot + self.global_mean_ + bias
            else:
                prediction[index] = float('NaN')
        
        return prediction

    
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