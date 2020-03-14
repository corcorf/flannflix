"""
Perform a specified number of recommender iterations to see what the
distribution of recommendations looks like
"""
import recommender
import pandas as pd
import numpy as np
from datetime import datetime
from time import time
from sqlalchemy import create_engine
from sql_tables import pick_random_movies

def test_recommender(engine, n_movies=5, n_recommendations=5, n_iterations=5000):
    """
    Perform a specified number of recommender iterations to see what the
    distribution of recommendations looks like
    Parameters:
        engine (sqlalchemy engine object):
        n_movies (int): number of movies to select for each iteration
        n_recommendations: number of recommendations that should be made in each iteration
        n_iterations: number of recommendation iterations that should be performed
    Returns
        pandas dataframe containing the randomly selected movies and corresponding
        recommendations for every iteration
    """
    Session = sessionmaker(bind=engine)
    session = Session()

    movies, ratings, rating_matrix, model = recommender.prep_for_recommendations(fill_value='median')
    recommendation_records = pd.DataFrame()

    for i in range(n_iterations):
        selections = pick_random_movies(session, n_movies)
        result = recommender.get_recommendations(selections, movies, ratings, rating_matrix, model, n_recommendations)
        result, interpreted_choices = result
        df = pd.DataFrame({"iteration":i,
                           "user_selections":selections,
                           "interpretations":interpreted_choices,
                           "recommendations":result})

        recommendation_records = recommendation_records.append(df)

    return recommendation_records


if __name__ = "__main__":
    from sql_tables import HOST, PORT, USERNAME, PASSWORD, DB
    conn_string = f'postgres://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DB}'
    engine = create_engine(conn_string, echo=False)

    start_time = time()
    recommendation_records = test_recommender(engine, n_movies=5, n_recommendations=5, n_iterations=5000)
    print("performed {} iterations in {} seconds".format(n_iterations, time() - start_time))

    fn = "test_results_{}.csv".format(datetime.now().strftime("%d%m%Y_%H%M"))
    recommendation_records.to_csv(fn)
