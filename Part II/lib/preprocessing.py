"""
This file defines all functions needed to perform preprocessing of the data.
"""

# Imports
import pandas as pd

from sklearn.model_selection import train_test_split
from lib import utils
import scipy.sparse as sparse
import numpy as np


def load_data(path):
    """
    This function loads data from a csv file
    Args:
        - path: string, the path to the .csv data file
        This file should contain a table with 4 columns
    Output:
        - ratings: panda DataFrame,a panda DataFrame object containing the 3
        first columns of the data file, with column names 'user', 'item',
        'rating'
    """
    ratings = pd.read_csv(path)
    nb_col = ratings.columns.size
    if nb_col != 4 and nb_col != 3:
        raise Exception("Frame has {} cols instead of 3 or 4".format(nb_col))
    elif nb_col == 4:
        ratings = ratings.drop([list(ratings)[3]], axis = 1)
    ratings.columns = ['user', 'item', 'rating']
    return ratings

def sample_data(path, random_state):
    """
    This function samples the data from the original ratings to a little over 
    1 M ratings
    Args:
        - path: string, the path to the .csv data file
        - random_state: int, random seed for sampling
    Output:
        - sampled_data : pandas Dataframe, containing the sampled ratings
    """
    data = load_data("ratings.csv")
    _, data_to_sample = separate_elements_with_few_ratings(3, data, "user")
    sampled_data = data_to_sample.sample(n = (int(1e6) + int(1e5)),
                                         random_state = random_state)
    _, rest_data = separate_elements_with_few_ratings(1, sampled_data, "user")
    rest_data.reset_index(drop = True, inplace = True)
    return rest_data


def count_ratings(dataframe, element):
    """
    This function counts for all values in column 'element' of a dataframe, the
    number of time they appear in the column
    Args:
        - dataframe: a pandas dataframe.
        - element: string, must correspond to the name of a column in dataframe
        We will count for that element category how many ratings
        it is involved in.
    Output:
        - keys: pandas Series, indexed on the elementId and containing 
        the counts of ratings per element
    """
    if not(element in dataframe.columns):
        raise Exception("No {} column in the dataframe".format(element))
    else:
            keys = dataframe[element].value_counts()
    return keys

def separate_elements_with_few_ratings(min_ratings, ratings, element):
    """
    This function isolates the ratings corresponding to items that have less
    than 'min_ratings' ratings and returns on one hand the ratings frame
    corresponding only to those and the ratings frame corresponding to all
    others
    Args:
        - min_ratings: int
        - ratings: a pandas dataframe
        - element: one of "item" or "element"
    Output:
        - t : a pandas dataframe with ratings corresponding to elements that
        have only one rating
        - reduced_ratings : a pandas dataframe, equal to ratings \ t
    """
    r_per_item = count_ratings(ratings, element)
    
    r_per_item = pd.DataFrame(r_per_item)
    r_per_item.columns = ['count']
    r_per_item[element] = r_per_item.index
    items_few_ratings = r_per_item[r_per_item['count'] <= min_ratings][element]
    t = ratings[ratings[element].isin(items_few_ratings)]
    reduced_ratings = ratings[-ratings[element].isin(items_few_ratings)]
    
    return t, reduced_ratings
    
def custom_full_train_test_split(data, random_state):
    """
    Takes in the whole data and divides it into train and test data
    with an approximate ratio of 0.85 training data
    Args:
        - data: a pandas dataframe
        - random_state: int, the random seed for the sampling
    Output:
        - final_train: a pandas dataframe, the train set
        - final_test: a pandas dataframe, the test set
    """
    train_1, rest_data = separate_elements_with_few_ratings(1, data, "item")
    train_2, test, _, _ = train_test_split(rest_data, 
                                           rest_data['item'],
                                           test_size = 0.2, 
                                           stratify = rest_data['item'],
                                           random_state = random_state)
    train_3, rest_test = separate_elements_with_few_ratings(2, test, "user")
    ratio = utils.unique_users(rest_test) / rest_test['user'].size
    final_test, train_4, _, _ = train_test_split(rest_test, 
                                                 rest_test['user'],
                                                 test_size = ratio*6, 
                                                 stratify = rest_test['user'],
                                                 random_state = random_state)
    final_train = pd.concat([train_1, train_2, train_3, train_4])
    return final_train, final_test

def custom_sampled_train_test_split(data, random_state):
    """
    Takes in the whole data and divides it into train and test data
    with an approximate ratio of 0.85 training data
    Args:
        - data: a pandas dataframe
        - random_state: int, the random seed for the sampling
    Output:
        - final_train: a pandas dataframe, the train set
        - final_test: a pandas dataframe, the test set
    """
    train_1, test, _, _ = train_test_split(data, 
                                           data['user'],
                                           test_size = 0.25, 
                                           stratify = data['user'],
                                           random_state = 0)
    train_2, rest_test = separate_elements_with_few_ratings(1, test, "item")
    ratio = utils.unique_movies(rest_test) / rest_test['item'].size
    final_test, train_3, _, _ = train_test_split(rest_test, 
                                                 rest_test['item'],
                                                 test_size = ratio*9, 
                                                 stratify = rest_test['item'],
                                                 random_state = 0)
    final_train = pd.concat([train_1, train_2, train_3])
    return final_train, final_test

def custom_small_train_test_split(data, random_state):
    """
    Takes in the whole data and divides it into train and test data
    with an approximate ratio of 0.85 training data
    Args:
        - data: a pandas dataframe
        - random_state: int, the random seed for the sampling
    Output:
        - final_train: a pandas dataframe, the train set
        - final_test: a pandas dataframe, the test set
    """
    t, reduced_ratings = separate_elements_with_few_ratings(1, data, "item")
    train, test,_,_ = train_test_split(reduced_ratings, 
                                       reduced_ratings['item'], 
                                       test_size = 0.15, 
                                       stratify = reduced_ratings['item'],
                                       random_state = 596)
    train = pd.concat([pd.DataFrame({'old_user' : [7026], 'old_item' : [590],
                                   'user': [254], 'item' : [24], 
                                   'rating' : [4.0]}), train], 
                    ignore_index = True)
    final_test = test[-(np.logical_and(test['user'] == 254, test['item'] == 24))]
    final_train = pd.concat([t, train])
    return final_train, final_test

def create_sparse_matrix(data, tot_items, tot_users):
    """
    Creates the sparse matrix from the data pandas Dataframe
    Args:
        - data: a pandas Dataframe, containing the users and items (hashed) as
        well as their ratings
        - tot_items: int, total number of items
        - tot_users: int, total number of users
    Output:
        - sparse_item_vectors: csr_matrix, compressed sparse row matrix with
        ratings for all items
    """
    global sparse_item_vectors
    sparse_item_vectors = sparse.lil_matrix((tot_items, tot_users))
    
    def fill_matrix(x):
        item = x['item']
        user = x['user']
        global sparse_item_vectors
        sparse_item_vectors[item, user] = x['rating']
    
    data.apply(fill_matrix, axis = 1)
    
    sparse_item_vectors = sparse_item_vectors.tocsr()
    
    return sparse_item_vectors


