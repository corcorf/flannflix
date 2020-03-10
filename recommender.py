import random
import pandas as pd
import numpy as np
import os
# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker

from sklearn.decomposition import NMF
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

from sql_tables import HOST, PORT, USERNAME, PASSWORD, DB
from sql_tables import Base, Movie, Rating, Tag, Link
from sql_tables import connect_to_db, read_tables


# MOVIES = ['Parasite', 'Moonlight', 'Star Wars', 'The Godfather', 'Das Boot', 'Pulp Fiction']

def get_rating_matrix(ratings_table, movies_table, movie_id_col="movieId",
                      user_id_col="userId", rating_col="rating"):
    """
    get a matrix of ratings vs movies
    """
    merged_with_ratings = ratings_table.merge(movies_table, on=movie_id_col, how="inner")
    rating_matrix = merged_with_ratings.set_index([user_id_col, movie_id_col])[rating_col].unstack(-1)
    return rating_matrix

def process_rating_matrix(rating_matrix, fill_value=0):
    """
    Process the ratings matrix by filling na values
    Returns new rating_matrix
    """
    return rating_matrix.fillna(fill_value)

def train_nmf_model(rating_matrix, n_components=20):
    """
    Returns trained NMF model
    """
    model = NMF(n_components=n_components)
    model.fit(rating_matrix)
    return model

def calculate_transformed_matrix(model, rating_matrix):
    """
    returns Rhat dataframe
    """
    R = rating_matrix
    Q = pd.DataFrame(model.components_, columns=R.columns)
    P = pd.DataFrame(model.transform(R), index=R.index)
    Rhat = pd.DataFrame(np.dot(P, Q), index=R.index, columns=R.columns)
    return Rhat

def get_movies_list(movies_df):
    """
    returns list of unique movies
    """
    return movies_df['title'].drop_duplicates().to_list()

def get_lookup_dict(df, key_column, value_column):
    """
    Returns look-up dictionary
    """
    return df.set_index(key_column)[value_column].to_dict()

def sanitize_titles(movie_choices, movies_df, rating_matrix):
    """
    Takes user-specified names of movies
    processes using fuzzywuzzy
    returns list of the closest matching titles in the database
    """
    all_movies = get_movies_list(movies_df)
    # convert titles using fuzzywuzzy
    movie_choices = [process.extractOne(title, all_movies)[0] for title in movie_choices]
    return movie_choices

def get_movieIds_from_titles(movie_choices, movies_df):
    """
    Takes list of sanitized movie titles and returns list of corresponding movieIds
    """
    lookup_dict = get_lookup_dict(movies_df, 'title', 'movieId')
    numbers = [lookup_dict[t] for t in movie_choices]
    return numbers

def create_user_choice_vector(rating_matrix, movie_choice_numbers, rating=5):
    """
    Takes movieIds of users selected movies
    Returns dataframe containing vector of users chosen movie with assumed rating
    """
    user_choice_vec = pd.DataFrame(index=rating_matrix.columns)
    user_choice_vec['rating'] = 0
    for n in movie_choice_numbers:
        user_choice_vec.loc[n,'rating'] = rating
    return user_choice_vec

def get_user_recommendation_vector(user_choice_vec, rating_matrix, model):
    """
    Returns dataframe containing predicted users ratings
    """
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
    recs = user_choice_vec.sort_values(ascending=False).head(n_recommendations)
    lookup_dict = get_lookup_dict(movies_df, 'movieId', 'title')
    return [lookup_dict[n] for n in recs.index]

def get_recommendations(movie_choices, n_recommendations=5, n_components=20, fill_value=0, conn_string=None):
    """
    returns list of recommendations
    """
    if conn_string == None:
        conn_string = f'postgres://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DB}'
    engine, session = connect_to_db(conn_string)

    data = read_tables(engine, table_names=['ratings','movies'])
    ratings = data['ratings']
    movies = data['movies']
    rating_matrix = get_rating_matrix(ratings, movies)

    rating_matrix = process_rating_matrix(rating_matrix, fill_value=fill_value)
    model = train_nmf_model(rating_matrix, n_components=n_components)
    Rhat = calculate_transformed_matrix(model, rating_matrix)

    movie_choices = sanitize_titles(movie_choices, movies, rating_matrix)
    numbers = get_movieIds_from_titles(movie_choices, movies)

    choice_vec = create_user_choice_vector(rating_matrix, numbers, rating=5)
    rec_vec = get_user_recommendation_vector(choice_vec, rating_matrix, model)
    recommendations = get_top_recommendations(n_recommendations, rec_vec.squeeze(), movies)
    return recommendations
