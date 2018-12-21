"""
This file contaings the `MF` class implementing our Matrix Factorization with
L2 regularization
"""

import numpy as np

from lib import utils



class MF():

    def __init__(self, n_epochs = 20, n_factors = 50,
                 learning_rate = .007, learning_rate_bias = .007,
                 lambd = 0.02,
                 unique_users = 0, unique_items = 0,
                 verbose = False):
        """
        This method is called when initializing the estimator and assigns
        values to the object's parameters. The default values where chosen to
        be close to the SVD defaults in the scikit `surprise` package
        Args:
            - self: MF object, to be initialized
            - n_epochs: int, the number of epochs
            - n_factors: int, the number of factors in the
            latent user and item matrices
            - learning_rate: float, the learning rate for the
            SGD procedure used to find the user and item matrices
            - learning_rate_bias: float, the learning rate for the
            SGD procedure used to find the user and item biases
            - lambd: float, the L2 regularization parameter
            chosen to be the same for user and item vectors
            - unique_users: int, the number of unique users
            - unique_items: int, the number of unique items
            - verbose: boolean, if True prints the progress of
            the fit procedure for training purposes
        """
        self.n_epochs = n_epochs
        self.n_factors = n_factors
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.learning_rate_bias = learning_rate_bias
        self.lambd = lambd
        self.unique_users = unique_users
        self.unique_items = unique_items

    def fit(self, X, y):
        """
        This methods fits the algorithm to the data
        Args:
            - self: MF object, to be fitted
            - X: pandas DataFrame, with hashed available pairs of item-user.
            The hashing is necessary to make sure items and users are
            numbered from 0 to unique_users/items, from practical purposes
            - y: pandas Series, with corresponding ratings to the pairs of
            item-user in X
        """

        # useful variables
        nb_users = self.unique_users
        nb_items = self.unique_items
        size = utils.total_ratings(X)

        # initialize the latent factor matrices
        U = np.random.rand(nb_users, self.n_factors)
        V = np.random.rand(nb_items, self.n_factors)

        # initialize the bias vectors
        Bu = np.zeros(nb_users)
        Bi = np.zeros(nb_items)

        # initialize errors
        err = np.zeros(size)

        # aliases
        lr = self.learning_rate
        lrb = self.learning_rate_bias
        l = self.lambd

        # compute global mean
        global_mean = np.mean(y)

        # initialize the visited users and items
        known_u = np.full(nb_users, False, dtype = bool)
        known_i = np.full(nb_items, False, dtype = bool)

        # run SGD to update the factors
        for epoch in range(self.n_epochs):

            if self.verbose and (epoch == 0 or (epoch + 1) % 10 == 0):
                print("Starting epoch {}".format(epoch + 1))

            # randomly shuffling training data
            random_indices = np.random.permutation(size)

            for index in random_indices:

                # retrieve user and item at the given index
                user = int(utils.get_user_at(index, X))
                item = int(utils.get_item_at(index, X))

                # mark user and items as visited
                known_u[user] = True
                known_i[item] = True

                # compute factor dot products
                dot_prod = np.dot(V[item,], U[user,])

                # compute bias contribution to rating
                biases = Bu[user] + Bi[item]

                # compute error for this rating
                err[index] = y[index] - (global_mean + biases + dot_prod)

                # update the biases
                Bu[user] += lrb * err[index]
                Bi[item] += lrb * err[index]

                # simultaneous update of the factors
                old_u = np.copy(U[user,])
                for f in range(self.n_factors):
                    U[user, f]+=lr * (err[index] * V[item, f] - l * U[user, f])
                    V[item, f]+=lr * (err[index] * old_u[f] - l * V[item, f])

        # store the learned results
        self.latent_users_ = U
        self.latent_items_ = V
        self.bias_user_ = Bu
        self.bias_item_ = Bi

        # store the training mean
        self.global_mean_ = global_mean

        # store visited items and users
        self.known_users_ = known_u
        self.known_items_ = known_i

        return self

    def predict(self, X):
        """
        This function predicts the ratings for the data in X if the algorithm
        has been fitted, and throws an error otherwise
        Args:
            - self: MF object, should have called fit method earlier
            - X: panda DataFrame, with hashed pairs of item-user without
            ratings
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

        # compute predictions
        for index in range(X['user'].size):

            user = utils.get_user_at(index, X)
            item = utils.get_item_at(index, X)

            bias_i = 0
            dot = 0
            bias_u = 0

            if(self.known_items_[item]):
                bias_i = self.bias_item_[item]
                if(self.known_users_[user]):
                    dot = np.dot(self.latent_items_[item,],
                             self.latent_users_[user,])
            if(self.known_users_[user]):
                bias_u = self.bias_user_[user]
            biases = bias_u + bias_i
            prediction[index] = dot + biases + self.global_mean_

        return prediction

    def score(self, X, y):
        """
        This function returns the score obtained at the end of training
        Args:
            - self: MF object, should have called fit method earlier
            - X: pandas DataFrame, with pairs of item-user with ratings
            - y: pandas Series, contains the ratings associated with pairs in X
        Output:
            - score: float, the score on the training data (- RMSE)
        """
        size = utils.total_ratings(X)
        prediction = self.predict(X)
        err = np.sum((y - prediction)**2)
        score = - np.sqrt(err / size)
        return score
