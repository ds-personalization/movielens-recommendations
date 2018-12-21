"""

"""

import pandas as pd

import matplotlib.pyplot as plt

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
    hashed_users = ratings['user'].rank(method = 'dense').astype(int)
    hashed_items = ratings['item'].rank(method = 'dense').astype(int)
    hashed_data = pd.DataFrame({'old_user' : ratings['user'],
                                'old_item' : ratings['item'],
                                'user' : hashed_users, 
                                'item' : hashed_items,
                                'rating' : ratings['rating']})
    return hashed_data

def plot_rmse(parameter_name, parameter_list, rmse_validation):
    """
    This function plots the values of rmse in 
    `rmse_validation` corresponding to the values in 
    `parameter_list`
    Args:
        - parameter_name: string, parameter for which different
        values were successively chosen to fit a model
        - parameter_list: list, list of values of the parameter
        - rmse_validation: list, list of values of the rmse
        after fitting a model for each value in parameter_list
    """
    plt.plot(parameter_list, rmse_validation, 'g^')
    plt.xlabel(parameter_name)
    plt.ylabel('RMSE')
    plt.title('Influence of {}s'.format(parameter_name))
    plt.show()
    
def print_table(rmse, rank_coeff, coverage):
    """
    This function prints the test results for the different metrics input
    Args:
        - rmse: dictionary, containing values of rmse associated with each
        algorithm
        - rank_coeff: dictionary, containing values of the Kendall rank 
        correlation coefficient associated with each algorithm
        - coverage: dictionary, containing values of coverage associated with
        each algorithm    
    """
    html_string = "<html><body><table>"
    html_string += "<tr><td><b>" + "Algorithms" + "</b></td><td><b>" + "RMSE" 
    html_string += "</b></td><td><b>" + "Kendall rank" + "</b></td><td><b>" 
    html_string += "Coverage" + "</b></td></tr>" + "<tr><td>" + "Baseline"
    html_string += "</td><td>" + str(rmse['Baseline']) + "</td><td>" 
    html_string += str(rank_coeff['Baseline']) + "</td><td>"
    html_string += str(coverage['Baseline']) + "</td></tr>" + "<tr><td>"
    html_string += "Matrix Factorization" + "</td><td>" + str(rmse['MF'])
    html_string += "</td><td>" + str(rank_coeff['MF']) + "</td><td>"
    html_string += str(coverage['MF']) + "</td></tr>" + "<tr><td>"
    html_string += "User-based NM" + "</td><td>" + str(rmse['NM'])
    html_string += "</td><td>" + str(rank_coeff['NM']) + "</td><td>"
    html_string += str(coverage['NM']) + "</td></tr>" +"</table></body></html>"

    return(html_string)