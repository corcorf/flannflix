import random
import pandas as pd
import dask.dataframe as dd
import numpy as np
import os
import logging
import pickle
# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker
from sklearn.decomposition import NMF
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from sql_tables import HOST, PORT, USERNAME, PASSWORD, DB
from sql_tables import Base, Movie, Rating, Tag, Link
from sql_tables import connect_to_db, read_tables
from time import time

logging.basicConfig(filename='debug.log', level=logging.DEBUG)

USE_DASK = os.getenv("USE_DASK")

logging.debug(f"USE_DASK = {USE_DASK}")

# class timer():
#     def __init__(self, function_name):
#         self.function_name = function_name
#         self.start_time = time()
#
#     def __enter__(self):
#         return self
#
#     def __exit__(self, exc_type, exc_value, traceback):
#         duration = time() - self.start_time
#         logging.debug(f"{self.function_name} ran for {duration} seconds")


def get_rating_matrix(ratings_table, movies_table=None, movie_id_col="movieId",
                      user_id_col="userId", rating_col="rating"):
    """
    get a matrix of ratings vs movies
    """
    logging.debug(f"calculating ratings matrix")
    # merged_with_ratings = ratings_table.merge(movies_table, on=movie_id_col, how="inner")
    # rating_matrix = merged_with_ratings.set_index([user_id_col, movie_id_col])[rating_col].unstack(-1)
    if USE_DASK:
        ratings_table = ratings_table.categorize(columns=[movie_id_col])
        rating_matrix = ratings_table.pivot_table(values=rating_col, index=user_id_col, columns=movie_id_col)
    else:
        rating_matrix = ratings_table.set_index([user_id_col,movie_id_col])[rating_col].unstack()
    return rating_matrix

def process_rating_matrix(rating_matrix, fill_value=0):
    """
    Process the ratings matrix by filling na values
    Returns new rating_matrix
    """
    logging.debug(f"processing ratings matrix by filling nans with {fill_value}")
    try:
        fill_value = float(fill_value)
        return rating_matrix.fillna(fill_value)
    except ValueError:
        if fill_value=='median':
            return rating_matrix.fillna(value=rating_matrix.quantile(0.5, axis=0))
        elif fill_value=='mean':
            return rating_matrix.fillna(value=rating_matrix.mean(axis=0))
        else:
            raise Exception("Could not interpret fill_value when processing rating matrix")


def train_nmf_model(rating_matrix, n_components=50):
    """
    Returns trained NMF model
    """
    logging.debug(f"training NMF model with {n_components} components")
    model = NMF(n_components=n_components)
    model.fit(rating_matrix)
    return model

def train_and_save_nmf_model(rating_matrix, n_components=50, filename='trained_model.pkl'):
    """
    Trains an NMF model and saves to file
    """
    model = train_nmf_model(rating_matrix, n_components=n_components)

    logging.debug(f"saving model to {filename}")
    binary = pickle.dumps(model)
    open(filename, 'wb').write(binary)

def load_trained_model(filename='trained_model.pkl'):
    """
    Load a previously trained model
    """
    logging.debug(f"loading model from file {filename}")
    binary = open(filename, 'rb').read()
    model = pickle.loads(binary)
    return model

def calculate_transformed_matrix(model, rating_matrix):
    """
    returns Rhat dataframe
    """
    logging.debug(f"calculating transformed ratings matrix")
    R = rating_matrix
    Q = pd.DataFrame(model.components_, columns=R.columns)
    P = pd.DataFrame(model.transform(R), index=R.index)
    Rhat = pd.DataFrame(np.dot(P, Q), index=R.index, columns=R.columns)
    return Rhat

def get_movies_list(movies_df):
    """
    returns list of unique movies
    """
    logging.debug(f"getting list of movies from pandas DataFrame object movies_df")
    if USE_DASK:
        movies_list = movies_df['title'].drop_duplicates().compute().to_list()
    else:
        movies_list = movies_df['title'].drop_duplicates().to_list()
    return movies_list

def get_lookup_dict(df, key_column, value_column):
    """
    Returns look-up dictionary
    """

    logging.debug(f"generating a lookup dictionary from df using keys from {key_column} and values from {value_column}")
    if USE_DASK:
        df = df[[key_column, value_column]].compute()
        return df.set_index(key_column)[value_column].to_dict()
    else:
        return df.set_index(key_column)[value_column].to_dict()

def interpret_titles(movie_choices, movies_df, rating_matrix):
    """
    Takes user-specified names of movies
    processes using fuzzywuzzy
    returns list of the closest matching titles in the database
    """
    logging.debug(f"interpreting titles with fuzzywuzzy")
    all_movies = get_movies_list(movies_df)
    # convert titles using fuzzywuzzy
    movie_choices = [process.extractOne(title, all_movies)[0] for title in movie_choices]
    return movie_choices

def get_movieIds_from_titles(movie_choices, movies_df):
    """
    Takes list of sanitized movie titles and returns list of corresponding movieIds
    """
    logging.debug(f"movies_df columns: {movies_df.columns}")
    lookup_dict = get_lookup_dict(movies_df, 'title', 'movieId')
    numbers = [lookup_dict[t] for t in movie_choices]
    return numbers

def create_user_choice_vector(rating_matrix, movie_choice_numbers, rating=5):
    """
    Takes movieIds of users selected movies
    Returns dataframe containing vector of users chosen movie with assumed rating
    """
    logging.debug(f"Creating user choice vector")
    user_choice_vec = pd.DataFrame(index=rating_matrix.columns)
    user_choice_vec['rating'] = 0
    for n in movie_choice_numbers:
        user_choice_vec.loc[n,'rating'] = rating
    return user_choice_vec

def get_user_recommendation_vector(user_choice_vec, rating_matrix, model):
    """
    Returns dataframe containing predicted users ratings
    """
    logging.debug(f"Getting user recommendation vector")
    new_user_id = rating_matrix.index.max() + 1
    R = rating_matrix
    Q = pd.DataFrame(model.components_, columns=R.columns)
    transformed_vec = model.transform(user_choice_vec.T)
    user_rec_vec = np.dot(transformed_vec, Q)
    user_rec_vec = pd.DataFrame(user_rec_vec, columns=R.columns, index=[new_user_id])

    return user_rec_vec.squeeze()

def add_user_to_rhat(user_choice_vec, rating_matrix, model, Rhat):
    """
    returns df of augmented Rhat
    """
    user_rec_vec = get_user_recommendation_vector(user_choice_vec, rating_matrix, model)
    augmented_rhat = pd.concat((Rhat, user_rec_vec))

    return augmented_rhat


def get_top_recommendations(n_recommendations, user_choice_vec, movies_df):
    """
    returns list of top recommendations for user
    """
    recs = user_choice_vec.sort_values(ascending=False).head(n_recommendations * 2)
    lookup_dict = get_lookup_dict(movies_df, 'movieId', 'title')
    recs = [lookup_dict[n] for n in recs.index]
    return recs

def remove_overlaps(recommendations, interpreted_choices):
    """
    Remove any of the recommendations that the user has seen
    """
    recs = [r for r in recommendations if r not in interpreted_choices]
    return recs


def prep_for_recommendations(n_components=20, fill_value=0, conn_string=None):
    """
    Set up tables needed for making recommendations
    """
    logging.debug(f"preparing data for movie recommendations")
    if conn_string == None:
        conn_string = f'postgres://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DB}'
    engine, session = connect_to_db(conn_string)

    data = read_tables(engine, table_names=['ratings','movies'])
    ratings = data['ratings']
    movies = data['movies']
    rating_matrix = get_rating_matrix(ratings, movies)

    rating_matrix = process_rating_matrix(rating_matrix, fill_value=fill_value)
    model = load_trained_model(filename='trained_model.pkl')
    # Rhat = calculate_transformed_matrix(model, rating_matrix)

    # all_the_stuff = {
    #     'model':model,
    #     'movies':movies,
    #     'ratings':ratings,
    #     'rating_matrix':rating_matrix,
    # }

    return movies, ratings, rating_matrix, model


def get_recommendations(movie_choices, movies, ratings, rating_matrix, model, n_recommendations=5):
    """
    returns a list of recommendations and a list of the interpreted choices
    """
    logging.debug(f"getting movie recommendations")
    interpreted_choices = interpret_titles(movie_choices, movies, rating_matrix)
    numbers = get_movieIds_from_titles(interpreted_choices, movies)

    choice_vec = create_user_choice_vector(rating_matrix, numbers, rating=5)
    rec_vec = get_user_recommendation_vector(choice_vec, rating_matrix, model)
    recommendations = get_top_recommendations(n_recommendations, rec_vec.squeeze(), movies)
    recommendations = remove_overlaps(recommendations, interpreted_choices)
    recommendations = recommendations[:n_recommendations]
    return recommendations, interpreted_choices
