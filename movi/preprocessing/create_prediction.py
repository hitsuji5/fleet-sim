import pandas as pd
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from db import engine, Session
from common.time_utils import get_local_datetime
from config.settings import MIN_LAT, MIN_LON, DELTA_LAT, DELTA_LON, MAP_WIDTH, MAP_HEIGHT


def create_predicted_demand(use_demand_pattern=False):
    request_table = "request_pattern" if use_demand_pattern else "request_backlog"
    query = "SELECT * FROM {}".format(request_table)
    df = pd.read_sql(query, engine, index_col="id")
    df["datetime_obj"] = df.request_datetime.apply(lambda x: get_local_datetime(x))
    df["dayofweek"] = df.datetime_obj.apply(lambda x: x.weekday())
    df["hour"] = df.datetime_obj.apply(lambda x: x.hour)
    df['day'] = df.datetime_obj.apply(lambda x: x.day)
    df['month'] = df.datetime_obj.apply(lambda x: x.month)
    # df["date"] = df.datetime_obj.apply(lambda x: x.strftime('%Y/%m/%d'))
    df['x'] = ((df['origin_lon'] - MIN_LON) / DELTA_LON).astype(int)
    df['y'] = ((df['origin_lat'] - MIN_LAT) / DELTA_LAT).astype(int)

    df = df.groupby(['month', 'day', 'hour', 'x', 'y']).size()
    df.name = 'demand'
    df = df.reset_index()

    table_name = "predicted_demand"
    drop_table = """
    DROP TABLE IF EXISTS {};
    """.format(table_name)
    Session.execute(drop_table)
    Session.commit()
    df.to_sql(table_name, engine, flavor=None, schema=None, if_exists='fail',
               index=True, index_label=None, chunksize=None, dtype=None)

    create_index = """
    CREATE INDEX index_predicted_demand ON predicted_demand (month, day, hour);
    """
    Session.execute(create_index)
    Session.commit()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", action='store_true', help = "use demand pattern")
    args = parser.parse_args()
    create_predicted_demand(args.pattern)
