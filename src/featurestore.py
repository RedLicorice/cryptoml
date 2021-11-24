from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.orm.session import Session
from config import config

engine = None
Base = declarative_base()
# Create missing tables
db_session = None

def get_engine_uri():
    return config['database']['sql']['uri'].get(str)

def init_engine(**kwargs):
    uri = get_engine_uri()
    if uri.startswith('sqlite://'):
      kwargs.update({'connect_args': {'check_same_thread': False}})
    engine = create_engine(uri, **kwargs)
    return engine


def save_df(storename: str, store):
    engine = init_engine()
    with engine.connect() as conn:
        store.to_sql(name=storename, con=conn, chunksize=config['database']['sql']['chunksize'].get(int), method = 'multi', if_exists='replace')


def load_df(name, features, begin=None, end=None):
    columns = ['timestamp']
    if type(features) is list:
         columns += features
    elif type(features) is str:
        columns += [features if not ',' in features else features.split(',')]
    elif not features:
        columns = ['*']
    else:
        raise ValueError('Features must either be a list or a comma-separed string')

    engine = init_engine()
    with engine.connect() as conn:
        import pandas as pd
        query = 'SELECT {columns} FROM {name}'.format(columns=','.join(columns), name=name)
        if begin or end:
            query += ' WHERE '
            if begin and end:
                query += 'timestamp >= {begin} AND timestamp <= {end}'.format(begin=begin, end=end)
            if begin and not end:
                query += 'timestamp >= {begin}'.format(begin=begin)
            if not begin and end:
                query += 'timestamp <= {end}'.format(end=end)
        query += ' ORDER BY timestamp ASC'

        return pd.read_sql_query(query, con=conn, chunksize=config['database']['sql']['chunksize'].get(int), parse_dates=True, index_col='timestamp')



