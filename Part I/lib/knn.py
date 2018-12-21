import numpy as np
from math import sqrt
from sklearn.base import BaseEstimator, RegressorMixin

from lib import utils

class knn(BaseEstimator, RegressorMixin):
    def __init__(self, n_neighbors = 10,
                unique_users = 0, unique_items= 0,
                sim_function = 'cosine',user_based = True,
                centered = False, standardized = False):
        """
        This method is called when initializing the model.
        Args:
            - self:
            - n_neighbors: int, number of neighbors to be considered
            - unique_users: int, number of unique users
            - unique_items: int, number of unique items
        """

        self.unique_users = unique_users
        self.unique_items = unique_items
        self.n_neighbors = n_neighbors
        self.sim_function = sim_function
        self.user_based = user_based
        self.centered = centered
        self.standardized = standardized
        self.RMSE = {}


    def compute_ranking_matrix(self, X, Y):
        """
        This function compute the ranking matrix with no changes made.
        Args:
            - X: pandas dataframe, created as "user", "item"
            - Y:pandas dataframe, with the corresponding "ratings"

        """

        # Useful variables
        unique_items = utils.unique_movies(X)
        unique_users = utils.unique_users(X)
        size = utils.total_ratings(X)

        # Setting new shape for knn model
        self.unique_items = unique_items
        self.unique_users = unique_users

        # Creating the ranking matrix
        M_bool = np.full((self.unique_users, self.unique_items), False)
        M_values = np.full((self.unique_users, self.unique_items), 0.0)

        for index in range(size):
            user = utils.get_user_at(index, X) - 1
            item = utils.get_item_at(index, X) - 1
            rating = Y[index]
            M_bool[user][item] = True
            M_values[user][item] = float(rating)

        self.matrix_bool = M_bool if self.user_based == True else np.transpose(M_bool)
        self.matrix_values = M_values if self.user_based == True else np.transpose(M_values)
        self.size = size

    def center_ranking(self):
        """
        This function computes the element mean ratings.
        """

        # aliases
        M_bool = self.matrix_bool
        M_values = self.matrix_values

        list_mean =[]
        for user in range(len(M_bool)):
            mean = np.mean(M_values[user][M_bool[user] == True])
            list_mean.append(mean)
            M_values[user][M_bool[user] == True]-= mean

        self.centered = True
        self.list_mean = list_mean

    def standardize_ranking(self):
        """
        This function standardizes the ranking matrix by user.
        """

        # aliases
        M_bool = self.matrix_bool
        M_values = self.matrix_values

        list_std = []
        for user in range(len(M_bool)):
            std = np.std(M_values[user][M_bool[user] == True])
            list_std.append(std)
            if std != 0:
                M_values[user][M_bool[user] == True]/= std

        self.standardize = True
        self.list_std = list_std

    def compute_sim(self, u_1, u_2):
        """
        Abstract method for computing the similarity between two users.
        According to the value of self.sim_function,
        it will call the corresponding method.
        """

        if self.sim_function == 'pearson':
            return self.pearson(u_1, u_2)
        elif self.sim_function == 'cosine':
            return self.cosine(u_1, u_2)
        elif self.sim_function == 'msd':
            return self.msd(u_1, u_2)

    def pearson(self, u_1, u_2):
        """
        This function computes the pearson correlation between two users.
        Args:
            - u_1: int, a user
            - u_2: int, another user
        return:
            - corr: float, the pearson correlation between u_1, u_2.
        """

        # aliases
        M_values = self.matrix_values

        corr = (np.dot(M_values[u_1], M_values[u_2]))**2
        temp = np.dot(M_values[u_1], M_values[u_2]) * np.dot(M_values[u_2], M_values[u_2])
        corr = corr / temp
        del temp

        return corr

    def cosine(self, u_1, u_2):
        """
        This function computes the cosine correlation between two users.
        Args:
            - u_1: int, a user
            - u_2: int, another user
        return:
            - corr: float, the cosine correlation between u_1, u_2.
        """

        # alias
        M_values = self.matrix_values

        corr = np.dot(M_values[u_1], M_values[u_2])
        temp = np.dot(M_values[u_1], M_values[u_1])
        temp*= np.dot(M_values[u_2], M_values[u_2])

        return corr / sqrt(temp)


    def msd(self, u_1, u_2):
        """
        This function computes the msd (Mean Squared Difference) similarity
        between two users.
        Args:
            - u_1: int, a user
            - u_2: int, another user
        return:
            - corr: float, the msd similarity between u_1, u_2.
        """
        pass
        #return corr

    def get_neighbors(self, u):
        """
        This function returns the index of the nearest neighbors of u.
        Args:
            -u: user to select the nearest neighbors from
        return:
            - neighbors: list, indexes of the nearest neighbors.
        """
        try:
            getattr(self, "sim_matrix")
        except AttributeError:
            raise RuntimeError("You must fit the model before trying to determine neighbors.")

        liste = self.sim_matrix[u]

        return sorted(range(len(liste)), key=lambda i: liste[i], reverse=True)[:self.n_neighbors]

    def predict_ranking(self, user, item, neighbors):
        """
        This function predicts an individual ranking of an item by a user given the neighbors.
        Args:
            - user: int, user
            - item: int, item
            - neighbors: list of ints, indexes of the nearest neighbors of user.
        """
        sum_sim = np.sum(self.sim_matrix[user][neighbors])
        if sum_sim == 0:
            pred = np.mean(self.matrix_values[user][self.matrix_bool[user] == True])
        else:
            temp = []
            for user in neighbors:
                val = self.matrix_values[user][item]
                if val >= 0:
                    temp.append(val)
                else:
                    temp.append(0.0)
            pred = np.dot(self.sim_matrix[user][neighbors],temp)
            pred = pred / sum_sim

        return pred

    def fit(self):
        """
        For knn, the fitting means the creation of the similarity matrix
        """

        M_sim = np.full((self.unique_users, self.unique_users), 0.0)
        for user_x in range(len(M_sim)):
            for user_y in range(user_x + 1, len(M_sim)):
                value = self.compute_sim(user_x, user_y)
                M_sim[user_x][user_y] = value
                M_sim[user_y][user_x] = value
        self.sim_matrix = M_sim

    def predict(self, X = None, y = None):
        """
        This function returns
        """
        try:
            getattr(self, "sim_matrix")
        except AttributeError:
            raise RuntimeError("You must fit the model before predicting.")

        # alias
        M_values = self.matrix_values
        M_bool = self.matrix_bool

        prediction = np.full((self.unique_users, self.unique_items), 0.0)

        for user in range(len(M_values)):
            neighbors = self.get_neighbors(user)
            for item in range(len(M_values[user])):
                if self.matrix_bool[user][item] == True:
                    prediction[user][item] = self.predict_ranking(user, item, neighbors)

        self.prediction = prediction
        return prediction

    def score(self, X=None, y=None):
        """
        This function computes the RMSE for a given value of k.
        """

        try:
            getattr(self, "prediction")
        except AttributeError:
            raise RuntimeError("You must have predicted the matrix before computing any score")
        if (self.n_neighbors not in self.RMSE.keys()):
            err = 0
            for user in range(len(self.matrix_values)):
                for item in range(len(self.matrix_values[user])):
                    if self.matrix_bool[user][item] == True:
                        err+= (self.prediction[user][item] - self.matrix_values[user][item])**2
            err = err / self.size

            dico = self.RMSE
            dico[self.n_neighbors] = err
            self.RSME = dico

    def gridSearch(self,k_values):
        """
        This method performs the gridSearch for knn.
        We decided to do our own gridSearch since we only need to fit once the data.
        We only need to compute once the similarity matrix, only the number of neighbors changes
        """
        dico = self.RMSE
        for k in k_values:
            self.n_neighbors = k
            self.predict()
            self.score()
        self.RMSE = dico
