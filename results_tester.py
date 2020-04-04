"""
Perform a specified number of recommender iterations to see what the
distribution of recommendations looks like
"""
from time import time
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import recommender
from sql_tables import pick_random_movies


def test_recommender(engine, n_movies=5, n_recommendations=5,
                     n_iterations=5000):
    """
    Perform a specified number of recommender iterations to see what the
    distribution of recommendations looks like
    Parameters:
        engine (sqlalchemy engine object):
        n_movies (int): number of movies to select for each iteration
        n_recommendations: number of recommendations that should be made in
                           each iteration
        n_iterations: number of recommendation iterations that should be
                      performed
    Returns
        pandas dataframe containing the randomly selected movies and
        corresponding recommendations for every iteration
    """
    session = sessionmaker(bind=engine)()

    _, _, rating_matrix, model =\
        recommender.prep_for_recommendations(fill_value='median')
    recommendation_records = pd.DataFrame()

    for i in range(n_iterations):
        selections = pick_random_movies(session, n_movies)
        result = recommender.get_recommendations(selections, rating_matrix,
                                                 model, n_recommendations)
        result, interpreted_choices = result
        record = pd.DataFrame({"iteration": i,
                               "user_selections": selections,
                               "interpretations": interpreted_choices,
                               "recommendations": result})

        recommendation_records = recommendation_records.append(record)

    return recommendation_records


if __name__ == "__main__":
    from sql_tables import HOST, PORT, USERNAME, PASSWORD, DB
    CONN_STRING = f'postgres://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DB}'
    ENGINE = create_engine(CONN_STRING, echo=False)
    N_ITERATIONS = 5000
    START_TIME = time()
    RECOMMENDATION_RECORDS = test_recommender(ENGINE, n_movies=5,
                                              n_recommendations=5,
                                              n_iterations=N_ITERATIONS)
    print("performed {} iterations in {} seconds".format(N_ITERATIONS,
                                                         time() - START_TIME))

    FILENAME = "test_results_{}.csv".format(
        datetime.now().strftime("%d%m%Y_%H%M")
    )
    RECOMMENDATION_RECORDS.to_csv(FILENAME)
