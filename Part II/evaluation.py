"""
This file contains the evaluation of our models
"""

# import necessary libraries
from lib import preprocessing
from lib import utils
from lib import Baseline as Bl
from lib import MF
from lib import MEMF_k_means as Mk
from lib import metrics
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

# split into train and test sets
if(full_data):
    train, test = preprocessing.custom_full_train_test_split(hashed_data,
                                                random_state = random_state)
elif(small_data):
    train, test = preprocessing.custom_small_train_test_split(hashed_data,
                                                random_state = random_state)
else:
    train, test = preprocessing.custom_sampled_train_test_split(hashed_data,
                                                random_state = random_state)

# create the hashing table used to retrieve true ids
hashing_table = pd.DataFrame({"old_item": train["old_item"],
                              "item": train["item"]})
    
# separate the ratings from the user-item pairs
y_train = train['rating']
y_test =  test['rating']

##-----------------------------------------------------------------------------

# TESTING

base = Bl.Baseline(tot_users, tot_items)
base.fit(train, y_train)

M = MF.MF(n_epochs = 20, 
       n_factors = 15,
       unique_users = tot_users, 
       unique_items = tot_items,
       verbose = False)
M.fit(train, y_train.values)

M1 = Mk.MEMF_k_means(n_clusters = 8,
                  tot_users = tot_users, 
                  tot_items = tot_items)
M1.generate_clusters_and_fit(hashing_table, train, y_train)

M2 = Mk.MEMF_k_means(n_clusters = 12,
                  n_clusters_prediction = 2,
                  tot_users = tot_users, 
                  tot_items = tot_items)
M2.generate_clusters_and_fit(hashing_table, train, y_train)


M3 = Mk.MEMF_k_means(n_clusters = 8,
                  n_clusters_prediction = 3,
                  fit_all = True,
                  tot_users = tot_users, 
                  tot_items = tot_items)
M3.generate_clusters_and_fit(hashing_table, train, y_train)

M_star = Mk.MEMF_k_means(n_clusters = 8,
                         n_clusters_prediction = 3,
                         fit_all = True,
                         dim_reduc = "MF",
                         tot_users = tot_users, 
                         tot_items = tot_items)
M_star.generate_clusters_and_fit(hashing_table, train, y_train)

#------
# RMSE


# Baseline
score_base = -base.score(test, y_test)
print(score_base)
# 0.9828

# MF
score_MF = -M.score(test, y_test.values)
print(score_MF)
# 0.9535

# Genre-MEMF



## TODO 



# K-means-MEMF-1

score_M1 = -M1.score(test, y_test.values, hashing_table)
print(score_M1)
# 0.9422

# K-means-MEMF-2

score_M2 = -M2.score(test, y_test.values, hashing_table)
print(score_M2)
# 0.9450

# K-means-MEMF-3

score_M3 = -M3.score(test, y_test.values, hashing_table)
print(score_M3)
# 0.9185

# K-means-MEMF*
scores_star = []
n_clusters_ = [3, 4, 6, 8, 10, 15, 20]
M_star = Mk.MEMF_k_means(n_clusters_prediction = 3,
                      fit_all = True,
                      dim_reduc = "MF",
                      tot_users = tot_users, 
                      tot_items = tot_items)
reduced_sparse_train = M_star.reduce_dimension(train)
for n_cluster in n_clusters_:
    M_star.set_n_cluster(n_cluster)
    M_star.compute_membership(hashing_table, reduced_sparse_train)
    M_star.group_by_cluster(hashing_table)
    M_star.fit(train, y_train)
    score = -M_star.score(test, y_test.values, hashing_table)
    scores_star.append(score)
    print(score)
# 0.9189 
# 0.9124 
# 0.9077
# 0.9061 
# 0.9070
# 0.9089 
# 0.9105


#------
# NDCG

ndcg = metrics.NDCG()
ndcg.create_ranking(train, test, y_train, y_test)

# Baseline
base_y_pred = base.predict(test)
base_ndcg = ndcg.NDCG(train, test, y_train, y_test, base_y_pred)
print(base_ndcg)


# MF
M_y_pred = M.predict(test)
M_ndcg = ndcg.NDCG(train, test, y_train, y_test, M_y_pred)
print(M_ndcg)

# Genre-MEMF



## TODO 




# K-means-MEMF-1
M1_y_pred = M1.predict(test, hashing_table)
M1_ndcg = ndcg.NDCG(train, test, y_train, y_test, M1_y_pred)
print(M1_ndcg)

# K-means-MEMF-2
M2_y_pred = M2.predict(test, hashing_table)
M2_ndcg = ndcg.NDCG(train, test, y_train, y_test, M2_y_pred)
print(M2_ndcg)

# K-means-MEMF-3
M3_y_pred = M3.predict(test, hashing_table)
M3_ndcg = ndcg.NDCG(train, test, y_train, y_test, M3_y_pred)
print(M3_ndcg)

# K-means-MEMF*
M_star_y_pred = M_star.predict(test, hashing_table)
M_star_ndcg = ndcg.NDCG(train, test, y_train, y_test, M_star_y_pred)
print(M_star_ndcg)

#-----------
# Diversity
    

# Baseline
base_cov = metrics.coverage(base, test, tot_users, tot_items)
print(base_cov)


# MF
M_cov = metrics.coverage(M, test, tot_users, tot_items)
print(M_cov)


# Genre-MEMF



## TODO 




# K-means-MEMF-1
M1_cov = metrics.coverage(M1, test, tot_users, tot_items)
print(M1_cov)

# K-means-MEMF-2
M2_cov = metrics.coverage(M2, test, tot_users, tot_items)
print(M2_cov)

# K-means-MEMF-3
M3_cov = metrics.coverage(M3, test, tot_users, tot_items)
print(M3_cov)

# K-means-MEMF*
M_star_cov = metrics.coverage(M_star, test, tot_users, tot_items)
print(M_star_cov)

    
#----------
# Coverage
    
# Baseline
base_cov = metrics.coverage(base, test, tot_users, tot_items)
print(base_cov)
# 0.3990

# MF
M_cov = metrics.coverage(M, test, tot_users, tot_items)
print(M_cov)
# 0.4012

# Genre-MEMF



## TODO 




# K-means-MEMF-1
M1_cov = metrics.coverage(M1, test, tot_users, tot_items, hashing_table, True)
print(M1_cov)
# 0.4027

# K-means-MEMF-2
M2_cov = metrics.coverage(M2, test, tot_users, tot_items, hashing_table, True)
print(M2_cov)
# 0.4025

# K-means-MEMF-3
M3_cov = metrics.coverage(M3, test, tot_users, tot_items, hashing_table, True)
print(M3_cov)
# 0.4027

# K-means-MEMF*
M_star_cov = metrics.coverage(M_star, test, tot_users, tot_items, 
                              hashing_table, True)
print(M_star_cov)
# 0.4036
    
    