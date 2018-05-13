import numpy as np
import pandas as pd
from db import engine
from common.time_utils import get_local_datetime
from config.settings import MAP_WIDTH, MAP_HEIGHT

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

        return demand


    def __compute_demand(self, x, d):
        return ((d[1] - d[0]) * x + (d[0] + d[1]) / 2) / 3600.0 * self.timestep * self.amplification_factor

    def update_hourly_demand(self, t):
        localtime = get_local_datetime(t - 60 * 30)
        current_time = localtime.month, localtime.day, localtime.hour
        if len(self.hourly_demand) == 0 or self.current_time != current_time:
            self.current_time = current_time
            demand1 = self.get_hourly_demand(t - 60 * 30)
            demand2 = self.get_hourly_demand(t + 60 * 30)
            demand3 = self.get_hourly_demand(t + 60 * 90)
            self.hourly_demand = [demand1, demand2, demand3]

        x = (localtime.minute - 30) / 60.0
        return x

    def get_hourly_demand(self, t):
        localtime = get_local_datetime(t)
        month, day, hour = localtime.month, localtime.day, localtime.hour
        query = """
          SELECT x, y, demand
          FROM predicted_demand
          WHERE month = {month} and day = {day} and hour = {hour};
                """.format(month=month, day=day, hour=hour)
        demand = pd.read_sql(query, engine, index_col=["x", "y"]).demand
        dmap = np.zeros((MAP_WIDTH, MAP_HEIGHT))
        for (x, y), c in demand.iteritems():
            dmap[x, y] += c
        return dmap

