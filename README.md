# Recommendation Project

__This is an ongoing project__

Authors : _Arthur Herbout_, _Redouane Dziri_

## Directories and files

Please look at the requirements file to learn about dependencies and useful packages to reproduce our results.

Directory __Part I__ contains the brute force baseline algorithms we developed as well as preliminary results.

Directory __Part II__ contains the more sophisticated recommender system we developed with our final results.

Both directories are independent and the code can be run independently from one or the other. Results and insights from __Part I__ are summed up in __Part II__ so that one can immediately look at the latter to understand our motivations, choices and results.

## Our goal

We are placing ourselves in the position of the owners of a movie streaming platform, akin to Netflix, Hulu, Showtime and the like.

We are going to **recommend movies to users this streaming platform** that have already rated at least one movie.

We wish to focus on a particular business objective : **recommend movies to users that are already watching at least a few movies on the platform**. Those users can be seen as valuable customers that are more likely to keep watching movies, especially if recommendations are relevant and, more importantly, are perceived as relevant by the user. Therefore we feel it is important to keep those users' business and enhance their experience with great recommendation to keep their interest up.

## The data

Our full dataset can be found here: [Full Movielens Dataset](https://grouplens.org/datasets/movielens/ )

## Quantitative results

See Notebooks in __Part I__ and __Part II__ folders.

## References and useful links

_surprise documentation_: 
Useful to get insights into the different baselines algorithms
https://surprise.readthedocs.io/en/stable/

_scikit-learn documentation_: 
Useful to know how to create class that can fit in the scikit-learn pipeline. 
For example, it enabled us to create our class that was able to use SearchGridCV from scikit
https://scikit-learn.org/stable/documentation.html

_Overview of Recommender Algorithms_:
Useful to get yet another vision of recommender systems
https://buildingrecommenders.wordpress.com/2015/11/18/overview-of-recommender-algorithms-part-2/

_"The Why and How of Nonnegative Matrix Factorization"_:
useful for future implementations of Matrix factorization
https://arxiv.org/pdf/1401.5226.pdf


