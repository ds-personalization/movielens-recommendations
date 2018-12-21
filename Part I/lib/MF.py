"""
This file contaings the `MF` class implementing our Matrix Factorization with
L2 regularization
"""

import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin

from lib import utils


class MF(BaseEstimator, RegressorMixin):  
     
    def __init__(self, n_epochs = 20, n_factors = 20, 
                 learning_rate = .007, learning_rate_bias = .007,
                 lambd = 0.02,
                 unique_users = 0, unique_items = 0,
                 verbose = False,
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
            
    def fit(self, X, y):
        """
        This methods fits the algorithm to the data
        Args:
            - self: MF object, to be fitted
            - X: panda DataFrame, with hashed available pairs of item-user
            - y: panda Series, with corresponding ratings to the
            pairs of item-user in X
        """
        
        # Useful variables
        nb_users = self.unique_users
        nb_items = self.unique_items
        size = utils.total_ratings(X)
        
        # Initialize the latent factor matrices
        U = np.random.rand(nb_users, self.n_factors)
        V = np.random.rand(nb_items, self.n_factors)
        
        # initialize the bias vectors
        Bu = np.zeros(nb_users)
        Bi = np.zeros(nb_items)
        
        # Initialize errors
        err = np.zeros(size)
        
        # Aliases
        lr = self.learning_rate
        lrb = self.learning_rate_bias
        l = self.lambd
        
        # Compute global mean
        global_mean = np.mean(y)
        
        # Initialize the visited users and items
        known_u = np.full(nb_users, False, dtype = bool)
        known_i = np.full(nb_items, False, dtype = bool)
        
        # Run SGD to update the factors
        for epoch in range(self.n_epochs):
            
            if self.verbose and (epoch == 0 or (epoch + 1) % 10 == 0):
                print("Starting epoch {}".format(epoch + 1))
            
            # randomly shuffling training data
            random_indices = np.random.permutation(size)
            
            for index in random_indices:
                
                # -1 because hashing created users and items with integer
                # values starting at 1
                user = utils.get_user_at(index, X) - 1
                item = utils.get_item_at(index, X) - 1
                
                known_u[user] = True
                known_i[item] = True
                
                # after experimentation with several methods to comput the dot
                # product we found the following to be the fastest
                dot_prod = np.dot(V[item,], U[user,]) 
                
                biases = Bu[user] + Bi[item]
                
                err[index] = y[index] - (global_mean + biases + dot_prod)
                
                # Update the biases
                Bu[user] += lrb * err[index]
                Bi[item] += lrb * err[index]
                
                # simultaneous update
                old_u = np.copy(U[user,])
                for f in range(self.n_factors):
                    U[user, f]+=lr * (err[index] * V[item, f] - l * U[user, f])
                    V[item, f]+=lr * (err[index] * old_u[f] - l * V[item, f])
                        
        # store the results
        self.latent_users_ = U
        self.latent_items_ = V
        self.bias_user_ = Bu
        self.bias_item_ = Bi
        
        # store the training mean
        self.global_mean_ = global_mean
        
        # store visited items and users
        self.known_users_ = known_u
        self.known_items_ = known_i
        
        # compute the score: -RMSE
        self.score_ = -np.sqrt(np.sum(err**2) / size)
        
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
            getattr(self, "latent_users_")
        except AttributeError:
            raise RuntimeError("You must fit the model before predicting.")
        
        # initialize predictions
        prediction = np.zeros(X['user'].size)
        
        for index in range(X['user'].size):
            
            user = utils.get_user_at(index, X) - 1
            item = utils.get_item_at(index, X) - 1
            
            if(self.known_users_[user] == True and 
               self.known_items_[item] == True):
                dot = np.dot(self.latent_items_[item,], 
                             self.latent_users_[user,]) 
                biases = self.bias_user_[user] + self.bias_item_[item]
                prediction[index] = dot + biases + self.global_mean_
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