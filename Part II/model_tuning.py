"""
This file contains the main personalization pipeline obtained to reproduce our
results
"""

# import necessary libraries
from lib import preprocessing
from lib import utils
from lib import MEMF_k_means as Mk
import pandas as pd

# initialize useful variables
full_data = False
small_data = False

random_state = 0

# read in the data
if(small_data):
    data = preprocessing.load_data("ratings_small.csv")
elif(full_data):
    data = preprocessing.load_data("ratings.csv")
else:
    data = preprocessing.sample_data("ratings.csv",random_state = random_state)

# compute the total number of ratings
tot_ratings = utils.total_ratings(data)

# compute the number of users
tot_users = utils.unique_users(data)

# compute the number of items
tot_items = utils.unique_movies(data)

# hash the data
hashed_data = utils.hash_data(data)

# split into train and validation sets
if(full_data):
    train, valid = preprocessing.custom_full_train_test_split(hashed_data,
                                                random_state = random_state)
elif(small_data):
    train, valid = preprocessing.custom_small_train_test_split(hashed_data,
                                                random_state = random_state)
else:
    train, valid = preprocessing.custom_sampled_train_test_split(hashed_data,
                                                random_state = random_state)

# create the hashing table used to retrieve true ids
hashing_table = pd.DataFrame({"old_item": train["old_item"],
                              "item": train["item"]})
    
# separate the ratings from the user-item pairs
y_train = train['rating']
y_validation =  valid['rating']

##-----------------------------------------------------------------------------

# MODEL TUNING

# K-means-MEMF-1

scores1 = []
n_clusters_ = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50]
M1 = Mk.MEMF_k_means(tot_users = tot_users, 
                     tot_items = tot_items)
reduced_sparse_train = M1.reduce_dimension(train)
for n_cluster in n_clusters_:
    M1.set_n_cluster(n_cluster)
    M1.compute_membership(hashing_table, reduced_sparse_train)
    M1.group_by_cluster(hashing_table)
    M1.fit(train, y_train)
    score = -M1.score(valid, y_validation.values, hashing_table)
    scores1.append(score)
    print(score)
    
# K-means-MEMF-2

scores2 = []
n_clusters_ = 7*[2, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 10, 10, 10]
n_clus_pred_ = 7*[2, 2, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4]
weights = 20*[1.] + 20*[1.5] + 20*[2.] + 20*[2.5] + 20*[3] + 20*[3.5] + 20*[4.]
M2 = Mk.MEMF_k_means(tot_users = tot_users, 
                     tot_items = tot_items)
reduced_sparse_train = M1.reduce_dimension(train)
for n_cluster, n_clus_pred, weight in zip(n_clusters_, n_clus_pred_, weights):
    M2.set_n_cluster(n_cluster)
    M2.set_n_cluster_prediction(n_clus_pred)
    M2.set_weight_penalty(weight)
    M2.compute_membership(hashing_table, reduced_sparse_train)
    M2.group_by_cluster(hashing_table)
    M2.fit(train, y_train)
    score = -M2.score(valid, y_validation.values, hashing_table)
    scores2.append(score)
    print(score)

# K-means-MEMF-3

scores3 = []
M3 = Mk.MEMF_k_means(fit_all = True,
                     tot_users = tot_users, 
                     tot_items = tot_items)
reduced_sparse_train = M3.reduce_dimension(train)
for n_cluster, n_clus_pred, weight in zip(n_clusters_, n_clus_pred_, weights):
    M3.set_n_cluster(n_cluster)
    M3.set_n_cluster_prediction(n_clus_pred)
    M3.set_weight_penalty(weight)
    M3.compute_membership(hashing_table, reduced_sparse_train)
    M3.group_by_cluster(hashing_table)
    M3.fit(train, y_train)
    score = -M3.score(valid, y_validation.values, hashing_table)
    scores3.append(score)
    print(score)

# K-means-MEMF*

scores4 = []
M4 = Mk.MEMF_k_means(fit_all = True,
                     dim_reduc = "MF",
                     tot_users = tot_users, 
                     tot_items = tot_items)
reduced_sparse_train = M4.reduce_dimension(train)
for n_cluster, n_clus_pred, weight in zip(n_clusters_, n_clus_pred_, weights):
    M4.set_n_cluster(n_cluster)
    M4.set_n_cluster_prediction(n_clus_pred)
    M4.set_weight_penalty(weight)
    M4.compute_membership(hashing_table, reduced_sparse_train)
    M4.group_by_cluster(hashing_table)
    M4.fit(train, y_train)
    score = -M4.score(valid, y_validation.values, hashing_table)
    scores4.append(score)
    print(score)