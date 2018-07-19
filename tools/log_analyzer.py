import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../movi/')
from dqn.settings import WORKING_COST, DRIVING_COST
from common import time_utils, vehicle_status_codes, customer_status_codes

log_dir_path = "data/logs/"
vehicle_log_file = "vehicle.log"
customer_log_file = "customer.log"
score_log_file = "score.log"
summary_log_file = "summary.log"

vehicle_log_cols = [
    "t",
    "id",
    "lat",
    "lon",
    "speed",
    "status",
    "destination_lat",
    "destination_lon",
    "assigned_customer_id",
    "time_to_destination",
    "idle_duration"
    ]

customer_log_cols = [
    "t",
    "id",
    "status",
    "waiting_time"
]

summary_log_cols = [
    "t",
    "n_vehicles",
    "occupied_vehicles",
    "n_requests",
    "n_matching",
    "n_dispatch"
]

score_log_cols = [
    "t",
    "vehicle_id",
    "earning",
    "idle",
    "cruising",
    "occupied",
    "assigned",
    "offduty"
]

class LogAnalyzer(object):

    def __init__(self):
        pass
        # if dir_path is None:
        #     dir_path = log_dir_path
        # self.log_dir_path = dir_path

    def load_log(self, path, cols, max_num, skip_minutes=0):
        df = pd.read_csv(path, names=cols)
        dfs = [df]
        for i in range(1, max_num):
            path_ = path + "." + str(i)
            try:
                df = pd.read_csv(path_, names=cols)
                dfs.append(df)
            except IOError:
                break
        df = pd.concat(dfs)
        df = df[df.t >= df.t.min() + skip_minutes * 60]
        return df

    def load_vehicle_log(self, log_dir_path, max_num=100, skip_minutes=0):
        return self.load_log(log_dir_path + vehicle_log_file, vehicle_log_cols, max_num, skip_minutes)

    def load_customer_log(self, log_dir_path, max_num=100, skip_minutes=0):
        return self.load_log(log_dir_path + customer_log_file, customer_log_cols, max_num, skip_minutes)

    def load_summary_log(self, log_dir_path, max_num=100, skip_minutes=0):
        return self.load_log(log_dir_path + summary_log_file, summary_log_cols, max_num, skip_minutes)

    def load_score_log(self, log_dir_path, max_num=100, skip_minutes=0):
        df = self.load_log(log_dir_path + score_log_file, score_log_cols, max_num, skip_minutes)
        df["t"] = (df.t - df.t.min()) / 3600
        df = df[df.t == df.t.max()]
        df["working"] = df.occupied + df.idle + df.cruising + df.assigned
        df["occupancy_rate"] = df.occupied / df.working
        df["cruising_rate"] = (df.cruising + df.assigned) / df.working
        df["working_rate"] = df.working / (df.working + df.offduty)
        df["reward"] = df.earning \
                       - (df.cruising + df.assigned + df.occupied) * DRIVING_COST \
                       - df.working * WORKING_COST
        return df

    def get_customer_status(self, customer_df, bin_width=300):
        customer_df["time_bin"] = self.add_time_bin(customer_df, bin_width)
        df = customer_df.groupby(["time_bin", "status"]).size().reset_index().pivot(index="time_bin", columns="status", values=0).fillna(0)
        df = df.rename(columns={2 : "ride_on", 4 : "rejected"})
        df["total"] = sum([x for _, x in df.iteritems()])
        df.index = [time_utils.get_local_datetime(x) for x in df.index]
        return df

    def get_customer_waiting_time(self, customer_df, bin_width=300):
        customer_df["time_bin"] = self.add_time_bin(customer_df, bin_width)
        df = customer_df[customer_df.status == 2].groupby("time_bin").waiting_time.mean()
        df.index = [time_utils.get_local_datetime(x) for x in df.index]
        return df

    def add_time_bin(self, df, bin_width):
        start_time = df.t.min()
        return ((df.t - start_time) / bin_width).astype(int) * int(bin_width) + start_time

