"""
This file aims at creating a global workflow for the genre.
We will first define  different methods:
    -  The basis one which consists in taking the genres given by the movies.csv file as ground truth.
This will serve as baseline.
    - The occurence similitude: this method also takes the genres given by the movies.csv file but performs some clustering.
     We will need to decide how to select the best number of clusters.
"""

import numpy as np
import pandas as pd

from lib import utils

class BaselineClustering():
    def __init__(self, genre_file = "/home/arthur/Documents/projects/Recommendation-Project/ml-latest/movies.csv"):
        """
        Method called when initializing the Clustering object

        """
        self.genre_file = genre_file
        self.data = pd.read_csv(self.genre_file)

    def find_clusters():
        """
        Base method that creates a dictionary of type:
            - key: movieId,
            - value: dictionary:
                - key : cluster
                - value: weight, measure of the membership of the movie to the cluster.
        In this first example, it will be a uniform weight.
        """
        movies = self.data.title.unique()
        clusters = {}
        for movie in movies:
            temp_dico = {}
            temp_data = self.data.loc[self.data['title'] == movie]
            size = temp_data.genre.size
            for index in range(0, size):
                genre = temp_data.iloc[index]['genre']
                temp_dico[genre] = 1 / size
            cluster[movie] = temp_dico

        self.clusters = clusters
