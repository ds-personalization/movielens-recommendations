"""

"""

import numpy as np
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

def predict_popular(model, X_train, X_test, y_test,
                    item = True,
                    threshold = 100,
                    ascending = False):
    """
    This method predict the ratings value for the top n_popular items.
    """
    # Generating the counts series
    X_test['rating'] = y_test
    if item == True:
        category1 = "old_item"
        category2 = 'old_item'
    else:
        category1 = "old_user"
        category2 = 'old_user'

    counts = X_train[category2].value_counts(ascending = ascending)
    counts = counts.to_frame()
    counts = counts.rename(index = str,
                            columns={category1: "count"})

    if ascending == False:
        counts = counts[counts['count'] >= threshold]
    else:
        counts = counts[counts['count']<= threshold]

    # creating a list of indexes
    indexes = counts.index.values

    # Now indexes is the list of old_item ids that have to be considered
    X_pop = X_test[X_test[category2].isin(indexes)]

    # Creating the corresponding y
    y_pop = X_pop.rating.values

    try:
      getattr(model, "tables")
      score = model.score(X_pop, y_pop)
    except AttributeError:
      score = model.score(X_pop, y_pop, X_train)
    return score

def predict_popular_exact(model, X_train, X_test, y_test,
                    item = True,
                    threshold = 100):
    """
    This method predict the ratings value for the top n_popular items.
    """
    # Generating the counts series
    X_test['rating'] = y_test
    if item == True:
        category1 = "old_item"
        category2 = 'old_item'
    else:
        category1 = "old_user"
        category2 = 'old_user'

    counts = X_train[category2].value_counts()
    counts = counts.to_frame()
    counts = counts.rename(index = str,
                            columns={category1: "count"})

    counts = counts[counts['count']== threshold]

    # creating a list of indexes
    indexes = counts.index.values

    # Now indexes is the list of old_item ids that have to be considered
    X_pop = X_test[X_test[category2].isin(indexes)]

    # Creating the corresponding y
    y_pop = X_pop.rating.values

    model.score(X_pop, y_pop)
    return model.scores


def score_distribution(model,  X_train, X_test, y_test,
                    item = True):
    """
    This method computes the distribution of scores accros the
    distribution of ratings per user or per item
    """
    X_test['rating'] = y_test
    if item == True:
        category1 = "old_item"
        category2 = 'old_item'
    else:
        category1 = "old_user"
        category2 = 'old_user'

    counts = X_train[category2].value_counts(ascending = True)
    counts = counts.to_frame()
    counts = counts.rename(index = str,
                            columns={category1: "count"})

    thresholds = counts['count'].unique()
    max_value = np.max(thresholds)

    # defining the distribution of scores
    scores = {}
    scores["counts"] = thresholds
    RMSE = []
    bias = []
    std_dev = []
    for threshold in thresholds:
        results = predict_popular_exact(model,
                                X_train, X_test, y_test, item, threshold)
        RMSE.append(results["RMSE"])
        bias.append(results["bias"])
        std_dev.append(results["standard deviation"])

    scores["RMSE"] = RMSE
    scores["bias"] = bias
    scores["std_dev"] = std_dev
    return scores


def total_ratings(ratings):
    """
    Args:
        - ratings: panda DataFrame, containing the user, item, and rating
        triplets
    Output:
        - total: int, the total number of ratings
    """
    total = ratings['user'].size
    return total

def unique_users(ratings):
    """
    Args:
        - ratings: panda DataFrame, containing the user, item, and rating
        triplets
    Output:
        - users: int, the total number of users
    """
    users = ratings['user'].unique().size
    return users

def unique_movies(ratings):
    """
    Args:
        - ratings: panda DataFrame, containing the user, item, and rating
        triplets
    Output:
        - movies: int, the total number of movies
    """
    movies = ratings['item'].unique().size
    return movies

def shape_matrix_index(ratings):
    """
    This function returns a positive float corresponding to the ratio of the
    number of movies to the number of users as a way to get an idea of how fat/
    square/thin the ratings matrix is
    We assume users are rows and movies are columns of the ratings matrix
    An output close to zero suggest that the ratings matrix is very thin while
    a large output (>> 1) suggests a fat matrix
    An output equal to 1 represents a square matrix
    Args:
        - ratings: panda DataFrame, containing the user, item, and rating
        triplets
    Output:
        - ratio: float, the ratio of number of movies to number of users
    """
    users = unique_users(ratings)
    movies = unique_movies(ratings)
    ratio = movies / users
    return ratio

def sparsity_index(ratings):
    """
    This function outputs a measure of the sparsity of the ratings matrix
    The output is a float between 0 and 1
    The closer the output is to 1, the more the matrix is sparse
    Conversely the closer the output is to 0, the more dense the matrix
    Args:
        - ratings: panda DataFrame, containing the user, item, and rating
        triplets
    Output:
        - sparsity: float, a measure of the sparsity of the matrix
    """
    users = unique_users(ratings)
    movies = unique_movies(ratings)
    data_points = total_ratings(ratings)
    sparsity = (users*movies - data_points)/(users*movies)
    return sparsity

def plot_distribution_ratings(dataframe):
    """
    This function counts, for all different values of ratings, their occurence
    and plots the results.
    Args:
        - dataframe: a panda dataframe.
    """

    keys = dataframe[dataframe.columns[2]].value_counts()
    keys.hist(bin = [0.5*i for i in range(9)])

def get_user_at(index, ratings):
    """
    This function returns the user in the ratings
    dataframe in a given row
    Args:
        - index: int, the index of the row
        - ratings: panda DataFrame ontaining the user, item, and rating
        triplets
    Output:
        - user: string/int, the value of user stored in ratings
        the type depends if the method is called before of after
        hashing users
    """
    user = ratings.iloc[index, ratings.columns.get_loc('user')]
    return user;

def get_item_at(index, ratings):
    """
    This function returns the item in the ratings
    dataframe in a given row
    Args:
        - index: int, the index of the row
        - ratings: panda DataFrame ontaining the user, item, and rating
        triplets
    Output:
        - item: string/int, the value of item stored in ratings
        the type depends if the method is called before of after
        hashing items
    """
    item = ratings.iloc[index, ratings.columns.get_loc('item')]
    return item;
  
def hash_data(ratings):
    """
    This functions is used to hash the users and items
    to integer values
    Args:
        - ratings: pd.DataFrame, containing the user-item 
        pairs whose ratings are available and their ratings
    Output:
        - hashed_data: pd.DataFrame, with hashed user - hashed item
        pairs whose ratings are available and their ratings
    """
    hashed_users = ratings['user'].rank(method = 'dense').astype(int) - 1
    hashed_items = ratings['item'].rank(method = 'dense').astype(int) - 1
    hashed_data = pd.DataFrame({'old_user' : ratings['user'],
                                'old_item' : ratings['item'],
                                'user' : hashed_users, 
                                'item' : hashed_items,
                                'rating' : ratings['rating']})
    return hashed_data

def get_true_user_at(index, ratings):
    """
    This function returns the user in the ratings
    dataframe in a given row as specified in the original dataset.
    Args:
        - index: int, the index of the row
        - ratings: panda DataFrame ontaining the user, item, and rating
        triplets
    Output:
        - user: string/int, the value of user stored in ratings
        the type depends if the method is called before of after
        hashing users
    """
    user = ratings.iloc[index, ratings.columns.get_loc('old_user')]
    return user;

def get_true_item_at(index, ratings):
    """
    This function returns the item in the ratings
    dataframe in a given row as specified in the original dataset.
    Args:
        - index: int, the index of the row
        - ratings: panda DataFrame ontaining the user, item, and rating
        triplets
    Output:
        - item: string/int, the value of user stored in ratings
        the type depends if the method is called before of after
        hashing users
    """
    item = ratings.iloc[index, ratings.columns.get_loc('old_item')]
    return item;
    
def hash_X_cluster(X):
    """
    This functions is used to hash the users and items
    to integer values
    Args:
        - X: pd.DataFrame, containing the user-item
        pairs whose ratings are available
    Output:
        - hashed_data: pd.DataFrame, with hashed user - hashed item
        pairs whose ratings are available and their ratings
    """
    hashed_users = X['user'].rank(method = 'dense').astype(int)
    hashed_items = X['item'].rank(method = 'dense').astype(int)
    hashed_X = pd.DataFrame({'old_user' : X['old_user'],
                                'old_item' : X['old_item'],
                                'user' : hashed_users,
                                'item' : hashed_items})
    return hashed_X


