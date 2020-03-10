import pandas as pd
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
import psycopg2

from sql_tables import Base, Movie, Rating, Tag, Link

HOST = 'localhost'
PORT = '5432'
USERNAME = 'flann'
PASSWORD = os.getenv('pgpassword')
DB = 'movies'
conn_string = f'postgres://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DB}'

def create_tables(conn_string):

    engine = create_engine(conn_string, echo=True)
    Session = sessionmaker(bind=engine)
    session = Session()

    logging.debug(f'Creating tables in sql')

    Base.metadata.create_all(engine)

    path = os.path.join(os.sep,'home','flann','spiced','data','ml-latest-small')

    # logging.debug(f'Writing to database')
    # conn = psycopg2.connect(conn_string)
    # cur = conn.cursor()
    # tables_files = {
    #     'ratings': os.path.join(path, 'ratings.csv'),
    #     'tags': os.path.join(path, 'tags.csv'),
    #     'links': os.path.join(path, 'links.csv'),
    #     'movies': os.path.join(path, 'movies.csv'),
    # }
    # for table_name, filename in tables_files.items():
    #     with open(fn, 'r') as f:
    #         next(f) # Skip the header row.
    #         cur.copy_from(f, table_name, sep=',')
    #         conn.commit()

    logging.debug(f'Loading data to database')
    ratings = pd.read_csv(os.path.join(path, 'ratings.csv'))
    tags = pd.read_csv(os.path.join(path, 'tags.csv'))
    links = pd.read_csv(os.path.join(path, 'links.csv'))
    movies = pd.read_csv(os.path.join(path, 'movies.csv'))

    movies.to_sql(name='movies',con=engine, if_exists='append', index=False,)
    ratings.to_sql(name='ratings',con=engine, if_exists='append', index=False,)
    tags.to_sql(name='tags',con=engine, if_exists='append', index=False,)
    links.to_sql(name='links',con=engine, if_exists='append', index=False,)

if __name__ == '__main__':
    create_tables(conn_string)
