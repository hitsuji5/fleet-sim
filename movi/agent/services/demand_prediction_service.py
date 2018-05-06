import numpy as np
import pandas as pd
from db import engine
from common.time_utils import get_local_datetime

class DemandPredictionService(object):

    def __init__(self, timestep=1800, amplification_factor=1.0):
        self.timestep = timestep
        self.amplification_factor = amplification_factor
        self.current_time = None
        self.hourly_demand = []

    def predict(self, t, horizon=1):
        x = self.update_hourly_demand(t)
        demand = []

        for _ in range(horizon):
            if abs(x) <= 0.5:
                d = self.__compute_demand(x, self.hourly_demand[0:2])
            elif 0.5 < x and x <= 1.5:
                d = self.__compute_demand(x - 1, self.hourly_demand[1:3])
            else:
                raise NotImplementedError

            x += self.timestep / 3600.0
            demand.append(d)

        return np.array(demand)


    def __compute_demand(self, x, d):
        return ((d[1] - d[0]) * x + (d[0] + d[1]) / 2) / 3600.0 * self.timestep * self.amplification_factor

    def update_hourly_demand(self, t):
        localtime = get_local_datetime(t - 60 * 30)
        dayofweek, hourofday = localtime.weekday(), localtime.hour
        if len(self.hourly_demand) == 0 or self.current_time != (dayofweek, hourofday):
            self.current_time = (dayofweek, hourofday)
            demand1 = self.get_hourly_demand(dayofweek, hourofday)

            localtime = get_local_datetime(t + 60 * 30)
            dayofweek, hourofday = localtime.weekday(), localtime.hour
            demand2 = self.get_hourly_demand(dayofweek, hourofday)

            localtime = get_local_datetime(t + 60 * 90)
            dayofweek, hourofday = localtime.weekday(), localtime.hour
            demand3 = self.get_hourly_demand(dayofweek, hourofday)

            self.hourly_demand = [demand1, demand2, demand3]

        x = (localtime.minute - 30) / 60.0
        return x

    def get_hourly_demand(self, dayofweek, hourofday):
        query = """
                          SELECT x, y, demand
                          FROM taxi.demand_trend
                          WHERE dayofweek = {dayofweek} and hourofday = {hourofday};
                            """.format(dayofweek=dayofweek, hourofday=hourofday)
        demand = pd.read_sql(query, engine, index_col=["x", "y"]).demand.to_dict()
        # demand = np.array([demand.get(road_id, 0) for road_id in RoadNetwork.road_ids])
        return demand

