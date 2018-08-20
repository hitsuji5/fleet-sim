import numpy as np
import pandas as pd
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from db import engine, Session
from common.time_utils import get_local_datetime
from config.settings import MIN_LAT, MIN_LON, DELTA_LAT, DELTA_LON, MAP_WIDTH, MAP_HEIGHT


def create_predicted_demand(source_table, target_table, variance_rate=None):
    query = "SELECT * FROM {}".format(source_table)
    df = pd.read_sql(query, engine, index_col="id")
    print("# of rows {}".format(len(df)))

    df["datetime_obj"] = df.request_datetime.apply(lambda x: get_local_datetime(x))
    df["dayofweek"] = df.datetime_obj.apply(lambda x: x.weekday())
    df["hour"] = df.datetime_obj.apply(lambda x: x.hour)
    df['day'] = df.datetime_obj.apply(lambda x: x.day)
    df['month'] = df.datetime_obj.apply(lambda x: x.month)
    df['x'] = ((df['origin_lon'] - MIN_LON) / DELTA_LON).astype(int)
    df['y'] = ((df['origin_lat'] - MIN_LAT) / DELTA_LAT).astype(int)

    df = df.groupby(['month', 'day', 'hour', 'x', 'y']).size()
    if variance_rate:
        df += np.maximum(-df.values, np.minimum(df.values, np.random.normal(0, np.sqrt(variance_rate * df.values))))
    df.name = 'demand'
    df = df.reset_index()

    drop_table = """
    DROP TABLE IF EXISTS {};
    """.format(target_table)
    Session.execute(drop_table)
    Session.commit()
    df.to_sql(target_table, engine, flavor=None, schema=None, if_exists='fail',
               index=True, index_label=None, chunksize=None, dtype=None)

    create_index = """
    CREATE INDEX index_{table} ON {table} (month, day, hour);
    """.format(table=target_table)
    Session.execute(create_index)
    Session.commit()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--variance", action='store', type=float)
    args = parser.parse_args()
    create_predicted_demand("request_backlog", "predicted_demand", args.variance)
