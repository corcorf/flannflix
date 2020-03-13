"""
Perform a specified number of recommender iterations to see what the
distribution of recommendations looks like
"""
import recommender
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from time import time
from sql_tables import connect_to_db, pick_random_movies
from sql_tables import HOST, PORT, USERNAME, PASSWORD, DB
logging.basicConfig(filename='debug.log', level=logging.DEBUG)

start_time = time()

conn_string = f'postgres://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DB}'
engine, session = connect_to_db(conn_string, verbose=False)

movies, ratings, rating_matrix, model = recommender.prep_for_recommendations(fill_value='median')

n_movies = 5
n_recommentations = 5
n_iterations = 5000

recommendation_records = pd.DataFrame()

for i in range(n_iterations):
    selections = pick_random_movies(session, n_movies)
    result = recommender.get_recommendations(selections, movies, ratings, rating_matrix, model, n_recommentations)
    result, interpreted_choices = result
    df = pd.DataFrame({"iteration":i,
                       "user_selections":selections,
                       "interpretations":interpreted_choices,
                       "recommendations":result})

    recommendation_records = recommendation_records.append(df)

print("performed {} iterations in {} seconds".format(n_iterations, time() - start_time))

fn = "test_results_{}.csv".format(datetime.now().strftime("%d%m%Y_%H%M"))
recommendation_records.to_csv(fn)
