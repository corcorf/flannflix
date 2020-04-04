"""
Module to create the database tables for the movie recommender and import the
MovieLens dataset from file
"""
import os
import logging
import pandas as pd
from sqlalchemy import create_engine
from sql_tables import Base

HOST = 'localhost'
PORT = '5432'
USERNAME = 'flann'
PASSWORD = os.getenv('pgpassword')
DB = 'movies'
CONN_STRING = f'postgres://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DB}'


def create_tables(conn_string):
    """
    create all tables in the sql database, if they don't already exist
    """
    engine = create_engine(conn_string, echo=True)
    logging.debug('Creating tables in sql')
    Base.metadata.create_all(engine)
    path = os.path.join(os.path.expanduser("~"),
                        'spiced', 'data', 'ml-latest-small')
    logging.debug('Loading data to database')
    ratings = pd.read_csv(os.path.join(path, 'ratings.csv'))
    tags = pd.read_csv(os.path.join(path, 'tags.csv'))
    links = pd.read_csv(os.path.join(path, 'links.csv'))
    movies = pd.read_csv(os.path.join(path, 'movies.csv'))

    movies.to_sql(name='movies', con=engine, if_exists='append', index=False)
    ratings.to_sql(name='ratings', con=engine, if_exists='append', index=False)
    tags.to_sql(name='tags', con=engine, if_exists='append', index=False)
    links.to_sql(name='links', con=engine, if_exists='append', index=False)


if __name__ == '__main__':
    create_tables(CONN_STRING)
