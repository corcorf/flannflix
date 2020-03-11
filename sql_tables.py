"""
Defines SQL tables for movie recommender
If run as main, will attempt to load data and create tables in database
Database must already exist
"""

import pandas as pd
import dask.dataframe as dd
import numpy as np
import os
import logging
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, ForeignKey, Integer, String, Boolean, DateTime, Float, BigInteger
from sqlalchemy import CheckConstraint
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from sqlalchemy.inspection import inspect
import psycopg2

USE_DASK = os.getenv("USE_DASK")
logging.debug(f"USE_DASK = {USE_DASK}")

Base = declarative_base()

HOST = 'localhost'
PORT = '5432'
USERNAME = 'flann'
PASSWORD = os.getenv('pgpassword')
DB = 'movies'
conn_string = f'postgres://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DB}'

class Movie(Base):
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

    def __repr__(self):
        return "<Movie(movie='%s', title='%s')>" % (
                            self.movieId, self.title)

class Rating(Base):
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
    userId = Column(BigInteger)#, primary_key=True)
    movieId = Column(BigInteger, ForeignKey('movies.movieId'))#, primary_key=True)
    rating = Column(Float)
    timestamp = Column(BigInteger)

    movie = relationship('Movie')

    def __repr__(self):
        return "<Rating(user='%s', movie='%s', rating='%s')>" % (
                            self.userId, self.movieId, self.rating)

class Tag(Base):
    """Class for new rows in tags SQL table
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
    userId = Column(BigInteger) #ForeignKey('ratings.userId'), , primary_key=True
    movieId = Column(BigInteger, ForeignKey('movies.movieId')) #, primary_key=True
    tag = Column(String)
    timestamp = Column(BigInteger)

    # ratings = relationship('Rating')
    movies = relationship('Movie')

    def __repr__(self):
        return "<Tag(id='%s', user='%s', movie='%s', tag='%s')>" % (
                            self.tagId, self.userId, self.movieId, self.tag)

class Link(Base):
    """Class for new rows in links SQL table
    _______
    columns:
        movieId (BigInteger)
        imdbId (BigInteger)
        tmdbId (BigInteger)
    """
    __tablename__ = "links"
    __table_args__ = ()

    movieId = Column(BigInteger, ForeignKey('movies.movieId'), primary_key=True)
    imdbId = Column(BigInteger)
    tmdbId = Column(BigInteger)

    movies = relationship('Movie')

    def __repr__(self):
        return "<Link(movie='%s', imdb='%s', tmdb='%s')>" % (
                            self.movieId, self.imdbId, self.tmdbId)

def get_table_name_from_class(class_name):
    """get a dictionary relating sqlalchemy ORM class names to SQL table names"""
    table_to_class = {c.__tablename__:c for c in [Rating, Tag, Movie, Link]}
    return table_to_class[class_name]

def connect_to_db(conn_string, verbose=False):
    """
    Connect to the database
    Returns an engine and a session
    """
    logging.debug(f"attempting to connect to DB with conn_string {conn_string}")
    engine = create_engine(conn_string, echo=verbose)
    Session = sessionmaker(bind=engine)
    session = Session()
    return engine, session

# def get_table_files(path=os.path.join(os.sep,'home','flann','spiced','data','ml-latest-small')):
#     """ NOT USED - part of attempt at a non-pandas method for loading to tables """
#     table_files = {
#         'movies': os.path.join(path, 'movies.csv'),
#         'tags': os.path.join(path, 'tags.csv'),
#         'links': os.path.join(path, 'links.csv'),
#         'ratings': os.path.join(path, 'ratings.csv'),
#     }
#     return table_files
#
# def process_files(table_files, conn_string):
#     """ NOT USED - part of attempt at a non-pandas method for loading to tables """
#     logging.debug(f'Writing to database')
#     conn = psycopg2.connect(conn_string)
#     cur = conn.cursor()
#     for table_name, filename in table_files.items():
#         with open(filename, 'r') as f:
#             next(f) # Skip the header row.
#             cur.copy_from(f, table_name, sep=',')
#             conn.commit()

def load_data(path=os.path.join(os.sep,'home','flann','spiced','data','ml-latest-small')):
    """load data from files"""
    logging.debug(f'Loading data to database')
    data = dict(
        movies = pd.read_csv(os.path.join(path, 'movies.csv')),
        ratings = pd.read_csv(os.path.join(path, 'ratings.csv')),
        tags = pd.read_csv(os.path.join(path, 'tags.csv')),
        links = pd.read_csv(os.path.join(path, 'links.csv')),
    )
    return data

def add_to_tables(data, engine):
    """add data in dictionary to sql engine"""
    logging.debug(f"adding data to SQL tables with connection {engine.url}")
    for name, df in data.items():
        df.to_sql(name=name,con=engine, if_exists='append', index=False)

def create_tables_and_add_data(conn_string):
    """
    Creates all tables in the sql database
    """
    logging.debug(f"Creating SQL tables")
    engine, session = connect_to_db(conn_string)
    logging.debug(f'Creating tables in sql')
    Base.metadata.create_all(engine)
    data = load_data()
    add_to_tables(data, engine)

# def alt_create_tables_and_add_data(conn_string):
#     """
#     NOT USED - Creates all tables in the sql database
#     """
#     engine, session = connect_to_db(conn_string)
#     logging.debug(f'Creating tables in sql')
#     Base.metadata.create_all(engine)
#     table_files = get_table_files()
#     process_files(table_files, conn_string)
#
# def select_all_from_table(session, table):
#     """
#     Select all from table in the postgres database
#     Returns a pandas dataframe
#     """
#     result = session.query(table).order_by(table.id)
#     return pd.DataFrame(result)
# def read_tables(engine, table_names = ['ratings', 'tags', 'links', 'movies']):
#     """Get pandas dataframes for all tables in table_names"""
#     all_tables = {}
#     for name in table_names:
#         all_tables[name] = pd.read_sql_table(name,engine)
#     return all_tables

def read_tables(engine, table_names = ['ratings', 'tags', 'links', 'movies']):
    """Get pandas dataframes for all tables in table_names"""
    logging.debug(f"reading SQL tables {table_names}")
    all_tables = {}
    if USE_DASK:
        for name in table_names:
            class_ = get_table_name_from_class(name)
            p_keys = [key.name for key in inspect(class_).primary_key][0]
            table = dd.read_sql_table(table=name, uri=engine.url, index_col=p_keys)
            all_tables[name] =  table
            logging.debug(f"df {name} has columns {table.columns}")
    else:
        for name in table_names:
            table = pd.read_sql_table(name,engine)
            all_tables[name] =  table
            logging.debug(f"df {name} has columns {table.columns}")

    return all_tables




if __name__ == '__main__':
    create_tables_and_add_data(conn_string)
