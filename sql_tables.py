"""
Defines SQL tables for movie recommender
If run as main, will attempt to load data and create tables in database
Database must already exist
"""

import os
import re
import logging
import pandas as pd
import numpy as np
import dask.dataframe as dd
from sqlalchemy import create_engine, distinct, CheckConstraint
from sqlalchemy import Column, ForeignKey, String, Float, BigInteger
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.inspection import inspect

logging.basicConfig(filename='debug.log', level=logging.DEBUG)

USE_DASK = os.getenv("USE_DASK") == "True"
logging.debug("USE_DASK = %s", USE_DASK)

BASE = declarative_base()

HOST = 'localhost'
PORT = '5432'
USERNAME = 'flann'
PASSWORD = os.getenv('pgpassword')
DB = 'movies'
CONN_STRING = f'postgres://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DB}'


class Movie(BASE):
    """
    Class for new rows in movies SQL table
    _______
    columns:
        movieId (BigInteger)
        title (String)
        genres (String)
    """
    __tablename__ = 'movies'

    movieId = Column(BigInteger, primary_key=True)
    title = Column(String)
    genres = Column(String)
    tokens = Column(TSVECTOR)

    def __repr__(self):
        return "<Movie(movie='%s', title='%s')>" % (self.movieId, self.title)


class Rating(BASE):
    """
    Class for new rows in ratings SQL table
    _______
    columns:
        ratingId (BigInteger)
        userId (BigInteger)
        movieId (BigInteger)
        rating (Float)
        timestamp (BigInteger)
    """
    __tablename__ = 'ratings'
    __table_args__ = (
        CheckConstraint('rating>=0'),
        CheckConstraint('rating<=5'),
        CheckConstraint('timestamp>=0'),
    )

    ratingId = Column(BigInteger, primary_key=True)
    userId = Column(BigInteger)
    movieId = Column(BigInteger, ForeignKey('movies.movieId'))
    rating = Column(Float)
    timestamp = Column(BigInteger)

    movie = relationship('Movie')

    def __repr__(self):
        return "<Rating(user='%s', movie='%s', rating='%s')>" \
            % (self.userId, self.movieId, self.rating)


class Tag(BASE):
    """
    Class for new rows in tags SQL table
    _______
    columns:
        tagId (BigInteger)
        userId (BigInteger)
        movieId (BigInteger)
        tag (String)
        timestamp (BigInteger)
    """
    __tablename__ = 'tags'
    __table_args__ = (
        CheckConstraint('timestamp>=0'),
    )

    tagId = Column(BigInteger, primary_key=True)
    userId = Column(BigInteger)
    movieId = Column(BigInteger, ForeignKey('movies.movieId'))
    tag = Column(String)
    timestamp = Column(BigInteger)

    movies = relationship('Movie')

    def __repr__(self):
        return "<Tag(id='%s', user='%s', movie='%s', tag='%s')>" \
            % (self.tagId, self.userId, self.movieId, self.tag)


class Link(BASE):
    """
    Class for new rows in links SQL table
    _______
    columns:
        movieId (BigInteger)
        imdbId (BigInteger)
        tmdbId (BigInteger)
    """
    __tablename__ = "links"
    __table_args__ = ()

    movieId = Column(BigInteger, ForeignKey('movies.movieId'),
                     primary_key=True)
    imdbId = Column(BigInteger)
    tmdbId = Column(BigInteger)

    movies = relationship('Movie')

    def __repr__(self):
        return "<Link(movie='%s', imdb='%s', tmdb='%s')>" \
            % (self.movieId, self.imdbId, self.tmdbId)


def get_table_name_from_class(class_name):
    """
    get a dictionary relating sqlalchemy ORM class names to SQL table names
    """
    table_to_class = {c.__tablename__: c for c in [Rating, Tag, Movie, Link]}
    return table_to_class[class_name]


def connect_to_db(conn_string, verbose=False):
    """
    Connect to the database
    Returns an engine and a session
    """
    logging.debug("attempting to connect to DB with conn_string %s",
                  conn_string)
    engine = create_engine(conn_string, echo=verbose)
    session = sessionmaker(bind=engine)()
    return engine, session


def load_data(path=os.path.join(os.sep, 'home', 'flann', 'spiced', 'data',
                                'ml-latest-small')):
    """load data from files"""
    logging.debug('Loading data to database')
    data = dict(
        movies=pd.read_csv(os.path.join(path, 'movies.csv')),
        ratings=pd.read_csv(os.path.join(path, 'ratings.csv')),
        tags=pd.read_csv(os.path.join(path, 'tags.csv')),
        links=pd.read_csv(os.path.join(path, 'links.csv')),
    )
    return data


def add_to_tables(data, engine):
    """add data in dictionary to sql engine"""
    logging.debug("adding data to SQL tables with connection %s", engine.url)
    for name, table in data.items():
        table.to_sql(name=name, con=engine, if_exists='append', index=False)


def create_tables_and_add_data(conn_string):
    """
    Creates all tables in the sql database
    """
    logging.debug("Creating SQL tables")
    engine, _ = connect_to_db(conn_string)
    logging.debug('Creating tables in sql')
    BASE.metadata.create_all(engine)
    data = load_data()
    add_to_tables(data, engine)


def read_tables(engine, table_names):
    """
    Get pandas dataframes for all tables in table_names
    """
    logging.debug("reading SQL tables %s", table_names)
    all_tables = {}
    if USE_DASK:
        for name in table_names:
            class_ = get_table_name_from_class(name)
            p_keys = [key.name for key in inspect(class_).primary_key][0]
            table = dd.read_sql_table(table=name, uri=engine.url,
                                      index_col=p_keys)
            all_tables[name] = table
            logging.debug("df %s has columns %s", name, table.columns)
    else:
        for name in table_names:
            table = pd.read_sql_table(name, engine)
            all_tables[name] = table
            logging.debug("df %s has columns %s", name, table.columns)

    return all_tables


def get_unique_movies(session):
    """Return an array of unique movie titles"""
    query = session.query(Movie).join(Rating).subquery()
    result = session.query(distinct(query.c.title)).all()
    result = np.array(result).reshape(-1).astype(str)
    return result


def get_movie_ids_and_titles(session):
    """Return a list of id, title tuples"""
    return session.query(Movie.movieId, Movie.title).all()


def query_movie_titles_from_ids(session, movie_ids):
    """query the database to get movie titles from ids"""
    query = session.query(Movie.movieId, Movie.title)
    query = query.filter(Movie.movieId.in_(movie_ids)).subquery()
    result = session.query(query.c.title).all()
    result = np.array(result).reshape(-1).astype(str)
    return result


def query_movie_ids_from_titles(session, movie_titles):
    """query the database to get movie id numbers from titles"""
    query = session.query(Movie.movieId, Movie.title)
    query = query.filter(Movie.title.in_(movie_titles)).subquery()
    result = session.query(query.c.movieId).all()
    result = np.array(result).reshape(-1).astype(int)
    return result


def get_lookup_dict(session, table_class, key_column, value_column):
    """Return a looup dictionary with keys and values from an sql table"""
    result = session.query(getattr(table_class, key_column),
                           getattr(table_class, value_column)).all()
    return dict(result)


def check_full_search_operator(text_search_op):
    """
    check that text_search_op is a valid operator for a postgres full-test
    search query. if not, break
    """
    operators = ['<->', '&', '|']
    if isinstance(text_search_op, str) and text_search_op.strip() in operators:
        pass
    else:
        raise Exception("unrecognised full text search operator")


def sanitize_search_term(search_string, regex_term=r"[^a-zA-Z\d\s]"):
    """
    remove characters from the search string that are likely to break the
    search
    """
    return re.sub(regex_term, " ", search_string)


def add_sql_full_search_operators(search_string, text_search_op="<->"):
    """
    add search operator to strings that contain spaces
    """
    return re.sub(r"\s+", f" {text_search_op} ", search_string.strip())


def full_text_search(session, table, id_column, token_column, search_string,
                     text_search_op="<->"):
    """
    perform a full text search query on a column containing text-search tokens
    return the corresponding value in the id_column
    """
    logging.debug("performing a full-text search for %s on the movies table",
                  search_string)

    search_string = sanitize_search_term(search_string)
    check_full_search_operator(text_search_op)
    search_string = add_sql_full_search_operators(search_string,
                                                  text_search_op)

    query = session.query(getattr(table, id_column),
                          getattr(table, token_column))
    query = query.filter(getattr(table, token_column).match(search_string))\
                 .subquery()
    result = session.query(getattr(query.c, id_column)).first()
    return result


def full_text_search_movies(session, search_string, text_search_op="<->"):
    """
    perform a full text search query on the titles in the movies table
    text_search_op is the operator that should be used in the postgres full
    text search
    return the first match of the title
    """
    logging.debug("performing a full-text search for %s on the movies table",
                  search_string)

    search_string = sanitize_search_term(search_string)
    check_full_search_operator(text_search_op)
    search_string = add_sql_full_search_operators(search_string,
                                                  text_search_op)

    query = session.query(Movie.movieId, Movie.title, Movie.tokens)
    logging.debug("sanitized search string is %s on the movies table",
                  search_string)
    query = query.filter(Movie.tokens.match(search_string)).subquery()
    result = session.query(query.c.title).first()
    if result is None:
        result = "Nothing to see here..."
    else:
        result = result.title
    return result


def pick_random_movies(session, n_movies=5):
    """
    generate a random list of movies chosen from the database
    """
    all_movies = get_unique_movies(session)
    movies = np.random.choice(all_movies, size=n_movies)
    return movies


if __name__ == '__main__':
    # create_tables_and_add_data(CONN_STRING)
    pass
