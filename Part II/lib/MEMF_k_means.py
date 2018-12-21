"""
This file contains the multi-entity algorithm with prior k-means clustering
"""

from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity

from lib.MF import MF
from lib import utils



class MEMF_k_means():
    
    
    
    def __init__(self, 
                 n_clusters = 20,
                 tot_items = 0,
                 tot_users = 0,
                 n_clusters_prediction = 1,
                 fit_all = False,
                 weight_penalty = 1.,
                 dim_reduc = "none",
                 dist = "",
                 verbose = False,
                 random_state = 0):
        """
        Method called when initializing the MEMF_k_means object
        Args:
            - n_clusters: int, number of clusters
            - tot_items: int, number of unique items
            - tot_users: int, number of unique users
            - n_clusters_prediction: int, the number of clusters each item will
            be assigned (with weights quantifying its membership)
            - fit_all: boolean, if True assigns items to all their clusters
            when fitting each cluster model; if False only assigns items to
            their main cluster when fitting each cluster model
            - weight_penalty: float, weight penalty used to compute membership
            of an item to the clusters
            - dim_reduc: string, can take three values "SVD", "MF" or "none"
            informs which dimensionality reduction technique is used in the
            preprocessing steps
            - dist: string, if "cosine" computes the cosine similarity between
            vectors when calculating membership to each cluster, else computes
            euclidian distance
            - verbose: boolean, regulates verbosity of training phase
            - random_state: int, random seed used by the k-means algorithm for
            initialization
        """
        self.n_clusters = n_clusters
        self.tot_items = tot_items
        self.tot_users = tot_users
        if(n_clusters_prediction > n_clusters):
            self.n_clusters_prediction = n_clusters
        else:
            self.n_clusters_prediction = n_clusters_prediction
        self.fit_all = fit_all
        self.weight_penalty = weight_penalty
        self.dim_reduc = dim_reduc
        self.dist = dist
        self.verbose = verbose
        self.random_state = random_state
        
    
    def set_n_cluster(self, n_cluster):
        self.n_clusters = n_cluster
        
    def set_weight_penalty(self, weight_penalty):
        self.weight_penalty = weight_penalty
    
    def set_n_cluster_prediction(self, n_cluster_prediction):
        self.n_clusters_prediction = n_cluster_prediction
    
    
    
    
    
    def reduce_dimension(self, data):
        """
        Transforms the data to either a sparse representation or a
        lower-dimension one depending on parameter `dim_reduc`
        If `dim_reduc` == "MF": carries out dimensional reduction using a
            matrix factorization fitted to the whole training data
        If `dim_reduc` == "SVD": carries out dimensional reduction using an
            SVD procedure fitted to the whole training data
        Else: transforms the data from a pandas DataFrame to a sparse matrix
        Args:
            - data: pandas DataFrame
        Output:
            - reduced_data: type depends on self.dim_reduc
            if "none" : csr_matrix
            if "SVD" or "MF": numpy ndarray
        """
        
        if(self.dim_reduc == "MF"):
            M = MF(n_epochs = 15, 
                   n_factors = 10,
                   unique_users = self.tot_users, 
                   unique_items = self.tot_items,
                   verbose = False)
            M.fit(data, data['rating'].values)
            reduced_data = M.latent_items_
            return reduced_data
       
        reduced_data = self.__make_sparse_matrix(data)
        
        if(self.dim_reduc == "SVD"):
            svd = TruncatedSVD(n_components = 20,
                               random_state = self.random_state)
            reduced_data = svd.fit_transform(reduced_data)
        
        return reduced_data
    
    
    def __make_sparse_matrix(self, data):
        """
        This function takes in a pandas ratings Dataframe and tranforms its 
        content to a scipy sparse matrix
        Args:
            - data: pandas DataFrame, containing (hashed!) user-item pairs and 
            associated ratings
        Output:
            - sparse_data: scipy csr_matrix, where each row represents a movie
        """
        data_frame = pd.pivot_table(data, 
                                    index = 'item', 
                                    columns = 'user', 
                                    values = 'rating')
        data_frame = pd.SparseDataFrame(data_frame)
        sparse_data = data_frame.to_coo()
        sparse_data = csr_matrix(sparse_data)
        return sparse_data
    
    
        

    def compute_membership(self, hashing_table, sparse_data):
        """
        Base method that creates a dictionary of type:
            - key: movieId,
            - value: dictionary:
                - key : cluster
                - value: weight, measure of the membership of the movie 
                to the cluster.
        Each movie will have as many clusters as `self.n_clusters_prediction`
        with the following weights:
            for a given movie m and a given cluster c
            let dist_c = distance between the movie and the centroid of c 
                    (either cosine or L2, depends on object parameter `dist`)
                Dist_c = exp(-weight_penalty * (dist_c / max(dist_c)))
                    where max(dist_c) is the maximum distance over all 
                    clusters of m (needed to avoid the exponential exploding)
                Dist_all = sum Dist_c for c in movie m's clusters
            weight_c = Dist_c / Dist_all
        Args:
            - sparse_data: csr matrix, containing the training data
            - hashing_table: a pandas dataframe, containing one column with 
            old item ids and one column for the hashed item ids
        """
        
        k_means = KMeans(n_clusters = self.n_clusters,
                         algorithm = 'full',
                         random_state = self.random_state)
        
        k_means.fit(sparse_data)

        self.centroids_ = k_means.cluster_centers_
        
        # store main clusters if don't want to fit all models
        if(not(self.fit_all)):
            self.main_cluster_per_item_ = k_means.labels_

        membership = {}
        
        # assign each item to `self.n_clusters_prediction` clusters
        for hashed_item in tqdm(range(self.tot_items)):
            
            # retrieve the item ratings
            item_row = self.__get_item_row_as_list(hashed_item, sparse_data)

            item_membership = {}
            
            # compute all non_normalized weights of membership of item to all
            # clusters
            all_weights = self.__compute_all_weights(item_row)
            
            # keep only the largest (ie. relevant) `self.n_clusters_prediction` 
            # weights and the indices of the clusters they refer to and 
            # normalize weights
            rel_indices, rel_weights = self.__keep_largest_weights(all_weights)
                
            for weight_index, cluster_index in enumerate(rel_indices):
                item_membership[cluster_index] = rel_weights[weight_index]

            original_item = self.__get_original_id(hashed_item, hashing_table)

            membership[original_item] = item_membership
        
        self.membership_ = membership
        
        
    
    
    def __get_item_row_as_list(self, item, sparse_data):
        """
        This function retrieves the item ratings at index `rating` in the 
        sparse matrix data and casts it as a list.
        Args:
            - item: int, index of an item
            - sparse_data: scipy csr_matrix, containing the (hashed) ratings 
            data
        Output:
            - item_row: list, containing all ratings for `item`
        """
        item_row = sparse_data[item]
        if(self.dim_reduc == "SVD" or self.dim_reduc == "MF"):
            return item_row.tolist()
        else:
            item_row = sparse_data.getrow(item)
            item_row = item_row.toarray()
            item_row = item_row[0].tolist()
        return item_row
    
    def __compute_all_weights(self, item_row):
        """
        This function computes all non-normalized membership weights for an
        item to the clusters
        Args:
            - item_row: list, all ratings for an item
        Output:
            - all_weights: np array, with non-normalized weights, one for each
            existing cluster
        """
        
        # initialize weights
        all_weights = np.zeros(self.n_clusters)
        
        # compute weights for each cluster
        for index, centroid in enumerate(self.centroids_):
            if self.dist == "cosine":
                dist_matrix = cosine_similarity([item_row, centroid])
            else:
                dist_matrix = euclidean_distances([item_row, centroid])
            dist = dist_matrix[0][1]
            all_weights[index] = dist
        
        # retrieve the largest distance
        max_dist = np.max(all_weights)
        
        # normalize before applying exponential and apply the function
        if(max_dist < float(1e-8)) :
            all_weights = np.ones(self.n_clusters)
        else:
            all_weights = all_weights / max_dist
            all_weights = np.exp(-self.weight_penalty * all_weights)
        
        return all_weights
    
    def __keep_largest_weights(self, all_weights):
        """
        This functions retrieves the `self.n_clusters_prediction` largest
        weights and their indices
        Args:
            - all_weights: np array, with weights for each cluster for a given
            movie
        Output:
            - relevant_indices: np array, with the indices of the largest 
            weights in the input array
            - relevant_weights: np array, with the largest weights in the input
            array
        """
        # we prefer the numpy argpartition to the numpy argsort method
        # as the first runs in linear time in the worst case, and we don't
        # care about ordering the largest weights
        relevant_indices = np.argpartition(all_weights, 
                                           -self.n_clusters_prediction)
        relevant_indices = relevant_indices[-self.n_clusters_prediction:]
        relevant_weights = all_weights.take(relevant_indices)
        relevant_weights = relevant_weights / np.sum(relevant_weights)
        return relevant_indices, relevant_weights
    
    def __get_original_id(self, hashed_item, hashing_table):
        """
        This function looks into the hashing table and retrieves the original
        item id
        Args:
            - hashed_item: int, hashed item id
            - hashing_table: pandas DataFrame, containing the hashing between
            original ids and new ids
        Output:
            - original_item: int, the original item id
        """
        original_item = hashing_table.loc[hashing_table['item'] == hashed_item,
                                      'old_item']
        original_item = original_item.iloc[0]
        return original_item
    
    
    
    
    

    def group_by_cluster(self, hashing_table, train = True):
        """
        That function creates a dictionary of pairs:
            - key: name of the genre
            - value: set of movieId that belong to that genre
        Args:
            - hashing_table: pandas DataFrame, containing the hashing function
            between original item ids and new ones
            - train: boolean, if True saves the cluster in the object's
            parameters; if False doesn't
        """
        
        clusters = dict((cluster, []) for cluster in range(self.n_clusters))

        if(self.fit_all):
            for original_item in tqdm(self.membership_.keys()):
                for clus_id in self.membership_[original_item]:
                    clusters[clus_id].append(original_item)
        else:
            # fill the dictionary with only one cluster per item
            for item, cluster in tqdm(enumerate(self.main_cluster_per_item_)):
                original_item = self.__get_original_id(item, hashing_table)
                clusters[cluster].append(original_item)
                
        # save it in in training phase
        if(train):
            self.clusters_ = clusters
        
        return clusters






    def fit(self, X, y):
        """
        This function needs to be run after define_cluster so that 
        self.clusters is defined. It trains as many models as there is 
        clusters. Since the fit method from MF is being called for each of 
        those models, we also need to keep trace of the transformation of the 
        user_id and item_id for each model. Indeed, the fit method from MF 
        relies on an indexation from 0 to n_unique item.
        Args:
            - X: pandas DataFrame, with the training data
            - y: pandas Series, with the associated ratings
        """
        
        try:
            getattr(self, "clusters_")
        except AttributeError:
            raise RuntimeError("Define the clusters before fitting the MEMF.")
            
        # initializes the dictionary containing the clusters' models
        models = {}
        
        # initializes the model's hashing tables for items
        tables = {}
        
        # fit each cluster's model
        for cluster in self.clusters_.keys():
            
            # retrieve data relevant to the cluster
            x_clus, y_clus = self.__filter_on_cluster_and_rehash(cluster, X, y)
            
            hashing_clus = pd.DataFrame({"old_item": x_clus["old_item"],
                                         "item": x_clus["item"]})
            tables[cluster] = hashing_clus
            
            unique_items = utils.unique_movies(x_clus)

            n_epochs, n_factors = self.__set_epochs_and_factors(unique_items)
            
            model_cluster = MF(unique_users = self.tot_users, 
                               unique_items = unique_items,
                               n_epochs = n_epochs,
                               n_factors = n_factors)
            model_cluster.fit(x_clus, y_clus)

            models[cluster] = model_cluster
            
            if(self.verbose):
                print("Cluster model {} has been fitted".format(cluster + 1))

        self.models_ = models
        self.tables_ = tables
        
        
        
    def __filter_on_cluster_and_rehash(self, cluster, X, y):
        """
        This function retrieves data for items in a given cluster (all users
        but only items in the clusters) as well as their associated ratings.
        It rehashes items so that their ids start at 0.
        Args:
            - cluster: int, the id of a cluster
            - X: pandas DataFrame, the training data (user-item pairs)
            - y: pandas Series, the training ratings
        Output:
            - X_clus: pandas DataFrame, all user-item pairs for items in the
            cluster (with re-hashed items)
            - y_clus: np array, associated ratings
        """
        cluster_items = self.clusters_[cluster]
        X_clus = X[X['old_item'].isin(cluster_items)].copy()
        # minus 1 so that hashing starts at 0 and not 1
        X_clus['item'] = X_clus['item'].rank(method = 'dense').astype(int) - 1
        clus_indexes = X_clus.index.values
        y_clus = y[clus_indexes].values
        return X_clus, y_clus
    
    def __get_hashing_table(self, X):
        """
        This function fetches the hashing table from the data
        Args:
            - X: pandas DataFrame, with item_user pairs (hashed)
        Output:
            - hashing_table: pandas DataFrame, hashing from original item ids
            to new ones
        """
        hashing_table = pd.DataFrame({"old_item": X["old_item"],
                                      "item": X["item"]})
        return hashing_table
    
    def __set_epochs_and_factors(self, unique_items):
        """
        This function computes the number of epochs and factors for the
        upcoming MF procedure. The motivation behind setting the number of 
        factors as functions of the number of items in a cluster was that 
        clusters with very few items require less parameters to explain their 
        ratings than clusters with a large number of items.
        The number of factors was chosen to be the following:
            max(1, int(np.exp(-5)*(unique_items**0.75)))
            why this function of `unique_items`?
                we were looking for a function which could be written
                as a*(unique_items)^(b)
                    and found a and b based on trial-and-error of fitting 
                    models to samples of our data
        Args:
            - unique_items: int, the number of items in the data to be fitted
        Output:
            - n_epochs: int, the number of epochs to feed the fitting procedure
            - n_factors: int, the number of factors to feed the fitting
            procedure
        """
        n_factors = max(1, int(np.exp(-5)*(unique_items**0.75)))
        n_epochs = 15
        return n_epochs, n_factors
        
        
        
        
    
    
    def generate_clusters_and_fit(self, hashing_table, X, y):
        """
        This high-level function combines the clustering and fitting phases to 
        generate prediction-ready models
        Args:
            - hashing_table: pandas DataFrame, contains the hashing between old
            item ids and new ones
            - X: pandas DataFrame, containing training user-item pairs
            - y: pandas Series, containing the associated ratings
        """
        X_sparse = self.reduce_dimension(X)
        self.compute_membership(hashing_table, X_sparse)
        self.group_by_cluster(hashing_table)
        self.fit(X, y)
        





    def predict(self, X, train_hashing):
        """
        This function predicts the ratings for the data in X
        if the algorithm has been fitted, and throws an error otherwise
        Its success is reliant on every items having been passed to the object
        during the clustering phase
        Args:
            - X: pandas DataFrame, with hashed pairs of item-user without 
            ratings
            - train_hashing: pandas DataFrame, contains the hashing of the
            training data
        Output:
            - prediction: np array, containing the predicted ratings for
            each user-item pair in X
        """
        
        try:
            getattr(self, "models_")
        except AttributeError:
            raise RuntimeError("You must fit the models before predicting")
        
        n_predict = X['user'].size
        prediction = np.zeros(n_predict)
        
        for index in tqdm(range(n_predict)):
            
            user = utils.get_user_at(index, X)
            original_item = utils.get_true_item_at(index, X)
            item_membership = self.membership_[original_item]
            pred = 0

            for cluster in item_membership.keys():
  
                model = self.models_[cluster]
                clus_weight = item_membership[cluster]
                    
                # act differently based on if each model was fitted or only the
                # main cluster
                if(self.fit_all):

                    pred_r = self.__get_prediction_from(cluster, train_hashing,
                                                      original_item, user,
                                                      model)
                    pred += clus_weight * pred_r
                # if only one model was fitted
                else:
                    # retrieve the latent factor prediction
                    # if cluster is the main cluster
                    th = train_hashing
                    item = th.loc[th['old_item'] == original_item, 'item']
                    item = item.iloc[0]
                    if(cluster == self.main_cluster_per_item_[item]):
                        pred_r = self.__get_prediction_from(cluster, 
                                                            train_hashing,
                                                            original_item, 
                                                            user,
                                                            model)
                        pred += clus_weight * pred_r
                    else:
                        bias = model.global_mean_
                        if(model.known_users_[user]):
                            bias += model.bias_user_[user]
                        pred += clus_weight * bias
                        
            prediction[index] = pred
        
        return prediction
        
        
                
    def __get_prediction_from(self, cluster, train_hashing, 
                              original_item, user, model):
        """
        Helper function to compute prediction
        Args:
            - cluster: int, the cluster of the input model
            - train_hashing: pandas DataFrame, the hashing of the training data
            - original_item: int, original item id
            - user: int, hashed user id
            - model: MF object, model associated to cluster `cluster`
        Output:
            - pred_rating: float, the prediction of the `model` of the `user`,
            `item` input pair
        """
        # retrieve cluster hashing table
        t = self.tables_[cluster]
    
        # retrieve the cluster hashing value for this item
        t_clus = train_hashing[train_hashing['old_item'].isin(t['old_item'])]
        i_clus = t.loc[t_clus['old_item'] == original_item,'item'].iloc[0]
        
        # predict the rating using the cluster model
        data = [[user, i_clus]]
        df = pd.DataFrame(data, columns= ['user', 'item'])
        pred_rating = model.predict(df)[0]
        
        return pred_rating
                
                
                
                
 
        
    def score(self, X, y, hashing_table):
        """
        This function returns the score obtained at the end of training
        Args:
            - self: MEMF object, should have called fit method earlier
            - X: panda DataFrame, with pairs of item-user with ratings
            - y: panda Series, contains the ratings associated with pairs in X
        Output:
            - score: float, the score on the training data (- RMSE)
        """
        size = utils.total_ratings(X)
        prediction = self.predict(X, hashing_table)
        prediction[prediction > 5] = 5
        prediction[prediction < 0] = 0
        err = np.sum((y - prediction)**2)
        score = - np.sqrt(err / size)
        return score
