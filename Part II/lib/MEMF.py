"""
This file aims at creating a global workflow for the genre.
We will first define  different methods:
    -  MEMF: the basis one which consists in taking the genres given by the movies.csv file as ground truth.
This will serve as baseline.
    - OccurenceMF. The occurence similitude: this method also takes the genres given by the movies.csv file but performs some clustering.
     We will need to decide how to select the best number of clusters.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm
import math
import scipy.stats as sc

from lib import utils
from lib.MF import MF

class MEMF_genres():
    def __init__(self, genre_file = 'genre_file', baseline = False):
        """
        Method called when initializing the MEMF object.
        Args:
            - self: MEMF object, to be initialized
            - genre_file: path to the file that contains all
            the information about movie genres
            - baseline: boolean value, whether we use the baseline version of
            MEMF. Default value is False

        """
        self.genre_file = genre_file
        self.data = pd.read_csv(self.genre_file)
        self.unique_items = self.data.title.unique().size
        self.baseline = False

    def compute_membership(self):
        """
        Base method that creates a dictionary of type:
            - key: movieId,
            - value: dictionary:
                - key : cluster
                - value: weight, measure of the membership of the movie
                to the cluster.
        In this first example, it will be a uniform weight.
        Args:
            - self: MEMF object, should have called define_clusters before
        Returns:
            - None
        """
        try:
            getattr(self, "clusters")
        except AttributeError:
            raise RuntimeError("You must define the clusters first.")

        movies = self.data.movieId.unique()
        membership = {}
        for movie in tqdm(movies):
            temp_dico = {}
            temp_data = self.data[self.data['movieId'] == movie]
            size = temp_data.genre.unique().size
            for index in range(0, size):
                genre = temp_data.iloc[index]['genre']
                temp_dico[genre] = 1 / size
            membership[movie] = temp_dico

        self.membership = membership

    def define_clusters(self):
        """
        That function returns a dictionary of pairs:
            - key: name of the genre
            - value: set of movieId that belong to that genre
        Args:
            - self: MEMF object
        Returns:
            - None
        """
        clusters = {}
        keys = self.data.genre.unique()
        for key in keys:
            value = self.data[self.data['genre'] == key].movieId.unique()
            clusters[key] = value
        self.clusters = clusters

        print("Number of Clusters: " + str(keys.size))


    def fit(self, X, y):
        """
        This function needs to be run after define_cluster so that
        self.clusters is defined.
        It trains as many models as there is clusters.
        Since the fit method from MF is being called for each of those models,
        we also need to keep trace of the transformation of the user_id and
        item_id for each model.
        Indeed, the fit method from MF relies on an indexation from 0 to
        n_unique item.
        Args:
            - self: MEMF object, should have called define_cluster before
            - X: panda dataframe, ratings dataframe
            - y: panda dataframe, true rating values

        """
        try:
            getattr(self, "clusters")
        except AttributeError:
            raise RuntimeError("You must define the clusters before fitting the MEMF.")

        models = {}
        tables = {}
        for cluster in self.clusters.keys():
            cluster_items = self.clusters[cluster]
            X_c = X[X['old_item'].isin(cluster_items)].copy()
            X_c['item'] = X_c['item'].rank(method = 'dense').astype(int) - 1

            indexes = X_c.index.values
            y_c = y[indexes]

            # defining the parameters of the MF model for each cluster
            unique_users = X['user'].unique().size
            unique_items = cluster_items.size
            n_factors_model = math.floor(2 * cluster_items.size**(1/3))

            if self.baseline == True:
                model_cluster = MF(unique_users = unique_users,
                 unique_items = unique_items, n_factors = n_factors_model,
                 n_epochs = 0)
            else:
                model_cluster = MF(unique_users = unique_users,
                unique_items = unique_items, n_factors = n_factors_model)
            model_cluster.fit(X_c, y_c)
            models[cluster] = model_cluster
            tables[cluster] = X_c
            print(str(cluster) + ' matrix factorization is over.')


        self.models = models
        self.tables = tables

    def predict(self, X, y = None):
        """
        This function predicts for a dataframe the associated ratings.
        Args:
            - self: MEMF object, should have called fit before.
            - X: panda dataframe, ratings dataframe.
            - y: None, only to be consistent with sklearn
        """
        try:
            getattr(self, "models")
        except AttributeError:
            raise RuntimeError("You must fit the models before predicting, one for each cluster.")

        # initialize predictions
        prediction = np.zeros(X['user'].size)

        for index in tqdm(range(X['user'].size)):

            user = utils.get_user_at(index, X)
            item = utils.get_true_item_at(index, X)
            item_membership = self.membership[item]

            pred = 0
            # only consider the clusters than are relevant to our item
            for cluster in item_membership.keys():
                # Getting back the dataframe restricted to the cluster
                table = self.tables[cluster].copy()

                model = self.models[cluster]
                weight = item_membership[cluster]

                # Finding item id in the cluster
                item_cluster = table[table['old_item'] == item].item.unique()[0]

                data = [[user, item_cluster]]
                # Creating a one row dataframe to feed the predict method
                df = pd.DataFrame(data, columns= ['user', 'item'])

                temp = model.predict(df)
                pred = pred + weight*temp[0]

            prediction[index] = pred

        return prediction

    def score(self,X, y):
        """
        That method compute the RMSE of our model when predicting
        on a given dataset.
        This function must be ran after the prediction.
        Args:
            - self: MEMF object, should have called predict method earlier
            - X: panda dataframe, predict will be run on it.
            - y: panda Series, contains the ratings associated with pairs
             in self.predict(X)
        Output:
            - self.score: python dictionary. We use the decomposition of
            RMSE in bias and variance. Consists of the keys:
                - "RMSE": float, the RMSE on the rating data
                - "bias": float, bias of our predictor
                - "standard deviation": float, standard deviation of
                our predictor
        """
        try:
            getattr(self, "models")
        except AttributeError:
            raise RuntimeError("You must fit the models before scoring it.")

        score = {}
        prediction = self.predict(X)

        # forbid negative and > 5 predicted values
        prediction[prediction > 5] = 5
        prediction[prediction < 0] = 0

        # computing scores
        score["RMSE"] = np.sqrt(np.sum((prediction - y)**2)/y.size)
        score["bias"] = np.sum(prediction - y)/y.size
        score["standard deviation"] = np.sqrt(np.sum((prediction - y)**2)/y.size - (np.sum(prediction - y)/y.size)**2)
        self.scores = score
        return score
