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

from lib import utils
from lib.MF import MF

class MEMF():
    def __init__(self, genre_file = 'genre_file'):
        """
        Method called when initializing the Clustering object

        """
        self.genre_file = genre_file
        self.data = pd.read_csv(self.genre_file)
        self.unique_items = self.data.title.unique().size

    def compute_membership(self):
        """
        Base method that creates a dictionary of type:
            - key: movieId,
            - value: dictionary:
                - key : cluster
                - value: weight, measure of the membership of the movie to the cluster.
        In this first example, it will be a uniform weight.
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
        This function needs to be run after define_cluster so that self.clusters is defined.
        It trains as many models as there is clusters.
        Since the fit method from MF is being called for each of those models,
        we also need to keep trace of the transformation of the user_id and item_id for each model.
        Indeed, the fit method from MF relies on an indexation from 0 to n_unique item.
        """
        try:
            getattr(self, "clusters")
        except AttributeError:
            raise RuntimeError("You must define the clusters before fitting the MEMF.")

        models = {}
        tables = {}
        for cluster in self.clusters.keys():
            cluster_items = self.clusters[cluster]
            X_cluster = X[X['old_item'].isin(cluster_items)].copy()
            X_cluster['item'] = X_cluster['item'].rank(method = 'dense').astype(int) - 1
            #X_cluster['user'] = X_cluster['user'].rank(method = 'dense').astype(int) - 1

            indexes = X_cluster.index.values
            y_cluster = y[indexes]

            unique_users = X['user'].unique().size
            unique_items = cluster_items.size
            model_cluster = MF(unique_users = unique_users, unique_items = unique_items)
            model_cluster.fit(X_cluster, y_cluster)
            models[cluster] = model_cluster
            tables[cluster] = X_cluster
            print(str(cluster) + ' matrix factorization is over')


        self.models = models
        self.tables = tables

    def predict(self, X, y = None):
        """

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

        self.prediction = prediction

    def score(self, y):
        """

        """
        try:
            getattr(self, "prediction")
        except AttributeError:
            raise RuntimeError("You must predict the models before scoring it.")

        self.RMSE = np.sqrt(np.sum((self.prediction - y)**2)/y.size)
        return self.RMSE

class OMF(MEMF):
    """
    OMF stands for Occurence Matrix Factorization.
    It only differs from MEMF by the weighting procedure.
    In MEMF, we choose an uniform weight for all genres that appear in the description of the movie.
    In OMF, we will weight more the genre that appear rarely in the dataset.
    Indeed, it means that this genre targets very well the movie.
    On the contrary, if a genre if assigned to almost every movie, then its relative importance will be reduced.
    In order to do that, we will use the exponential exp(-x) function.
    """
    def compute_membership(self):
        """
        Base method that creates a dictionary of type:
            - key: movieId,
            - value: dictionary:
                - key : cluster
                - value: weight, measure of the membership of the movie to the cluster.
        In this example, it will be a exponential weight.
        First we come up with a scaling parameter so that we only take exponential between 0 and 1.
        """
        try:
            getattr(self, "clusters")
        except AttributeError:
            raise RuntimeError("You must define the clusters before computing membership.")

        # factor will be the max of the cluster size
        factor = 0
        for cluster in self.clusters.keys():
            value = self.clusters[cluster].size
            factor = max(value, factor)
        self.factor = factor

        movies = self.data.movieId.unique()
        membership = {}
        for movie in tqdm(movies):
            temp_dico = {}
            temp_data = self.data[self.data['movieId'] == movie]
            genres = temp_data.genre.unique()
            denom = 0
            for cluster in genres:
                denom = denom + math.exp(-self.clusters[cluster].size / factor)
            size = genres.size
            for index in range(0, size):
                genre = temp_data.iloc[index]['genre']
                temp_dico[genre] = math.exp(-self.clusters[genre].size / factor) / denom
            membership[movie] = temp_dico

        self.membership = membership

class COMF(MEMF):
    """
    COMF stands for Co-Occurence Matrix Factorization.
    """
    def define_clusters(self):
        """

        """
        movies = self.data.movieId.unique()
        genres = self.data.genre.unique()
        genre_dictionary = {}
        movie_dictionary = {}

        count = 0
        for movie in movies:
            movie_dictionary[movie] = count
            count+=1

        count = 0
        for genre in genres:
            genre_dictionary[genre] = count
            count+=1

        matrix = np.zeros((movies.size, genres.size))
        for index, row in self.data.iterrows():
            genre = row['genre']
            movieId = row['movieId']
            matrix[movie_dictionary[movieId]][genre_dictionary[genre]] = 1

        #storing the matrix if needed
        self.matrix = matrix

        #initializing kmeans to have same number of clusters
        kmeans = KMeans(n_clusters=genres.size, random_state=0)
        new_genres = kmeans.fit_predict(matrix)

        clusters = {}
        for i in range(genres.size):
            clusters[i] = np.where(new_genres == i)[0]

        self.clusters = clusters
        self.new_genres = new_genres
        self.movie_dictionary = movie_dictionary

    def compute_membership(self):
        """

        """
        try:
            getattr(self, "clusters")
        except AttributeError:
            raise RuntimeError("You must define the clusters first.")

        movies = self.data.movieId.unique()
        membership = {}
        for movie in tqdm(movies):
            temp_dico = {}
            temp_dico[self.new_genres[self.movie_dictionary[movie]]] = 1
            membership[movie] = temp_dico

        self.membership = membership
