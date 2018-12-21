"""
This file defines all functions needed to perform preprocessing of the data.
"""

# Imports
import pandas as pd

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
    nb_col = ratings.column.size
    if nb_col != 4:
        raise Exception("Dataframe has {} columns instead of 4".format(nb_col))
    else:
        ratings = ratings.drop([list(ratings)[3]], axis = 1)
        ratings.columns = ['user', 'item', 'rating']
    return ratings

def count_ratings(dataframe, element):
    """
    This function counts for all values in column 'element' of a dataframe, the
    number of time they appear in the column
    Args:
        - dataframe: a panda dataframe.
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

def find_min_ratings(series, nb_elements):
    """
    This function takes in a numeric series and returns the maximum value
    (threshold) within this series such that there are approximately 
    'max_n_elements' in the series whose values are equal to or larger than
    that threshold using dichotomy 
    Args:
        - series: panda Series, with numeric values (int, float..)
        - nb_elements: int, number of elements wanted after the data is
        filtered on values higher than the threshold returned
    Output:
        - value: int, maximum value in the series that achieves a total
        of n_elements selected if we filter the series on values higher than
        'value'
        - n_elements_selected: int, number of elements that will actually be 
        selected - ideally will be 'nb_elements'
    """
    value = 1
    while series[series >= value].size > nb_elements:
        value = value * 2
    left = value / 2
    right = value
    temp = left
    current = series[series >= right].size
    while left != right:
        temp = left + (right - left) // 2
        if temp == left:
            break
        current = series[series >= temp].size
        if current > nb_elements:
            left = temp
        elif current < nb_elements:
            right = temp - 1
        else :
            break
    value = temp
    n_elements_selected = current
    return value, n_elements_selected

def filter_element(dataframe, element, n_element):
    """
    This function does the reduction of the dataframe on the column element
    so that we only keep the n most used elements.
    Args:
    - dataframe: panda dataframe to be filtered.
    - element: string, one of {"user", "item"}
    - n_element: int, number of elements to keep in the reduced dataframe
    Output:
    - filtered_data: panda dataframe consisting in only the ratings of items
    that have been rated at least n times.
    - n_element_real: int, number of element has actually remains
    in the dataframe
    """
    if element != "item" and element != "user":
        raise Exception("Argument must be one of {user,item}.")
    else:
        count_element = count_ratings(dataframe, element)
        threshold_element, n_real_element = find_min_ratings(count_element, 
                                                             n_element)
        df = pd.DataFrame([count_element])
        df = df.transpose()
        df.columns = ['count']
        df[element] = df.index
        filtrd_data = pd.merge(dataframe, df, on=element, how='outer')
        filtrd_data = filtrd_data[filtrd_data['count'] >= threshold_element]
        filtrd_data = filtrd_data.drop([list(filtrd_data)[3]], axis = 1)
    return filtrd_data, n_real_element

def reduce_dataframe(df, n_items, n_users):
    """
    This function does the reduction first in items then in users so that we
    are sure each user in the reduced dataframe has at least rated n_
    Args:
        - df: panda dataframe, to be reduced to n_items and n_users
        - n_items: int, number of items in the output
        - n_users: int, number of users in the output
    Output :
        - reduced_dataframe: panda dataframe, reduced to n_items_real items and
          n_users_real users
        - n_items_real: int, actual number of items in the reduced_dataframe
        - n_users_real: int, actual number of users in the reduced_dataframe
    l'un puis l'autre
    """
    df, _ = filter_element(df, "item", n_items)
    reduced_dataframe, n_users_real = filter_element(df, "user", n_users)
    n_items_real = count_ratings(reduced_dataframe, "item")
    return reduced_dataframe, n_items_real, n_users_real

def sampling_data(df, element, size, random_state = None):
    """
    This function samples data from the data frame by sampling on the _element_
    column to have _size_ distinct id in the returned _element_ column
    Args:
        - df: panda dataframe, to be reduced via sampling
        - element: string, one of {"user", "item"}
        - size: int, the number of ids in element to be sampled
    Output:
        - reduced_dataframe: panda dataframe, result of the sampling procedure
    """
    unique_element = df[element].unique()
    unique_element_series = pd.Series(unique_element)
    element_sample = unique_element_series.sample(n = size, 
                                                  random_state = random_state)
    reduced_dataframe = df[df[element].isin(element_sample.values)]
    return reduced_dataframe

def separate_items_with_few_ratings(min_ratings, ratings):
    """
    This function isolates the ratings corresponding to items that have less
    than 'min_ratings' ratings and returns on one hand the ratings frame
    corresponding only to those and the ratings frame corresponding to all
    others
    Args:
        -
    """
    r_per_item = count_ratings(ratings, "item")
    
    r_per_item = pd.DataFrame(r_per_item)
    r_per_item.columns = ['count']
    r_per_item['item'] = r_per_item.index
    items_few_ratings = r_per_item[r_per_item['count'] <= min_ratings]['item']
    t = ratings[ratings['item'].isin(items_few_ratings)]
    reduced_ratings = ratings[-ratings['item'].isin(items_few_ratings)]
    
    return t, reduced_ratings
    
