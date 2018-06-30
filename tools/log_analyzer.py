import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../movi/')
from config.settings import START_TIME, TIMESTEP
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
    "earning",
    "idle",
    "occupied",
    "cruising",
    "assigned",
    "offduty"
]

class LogAnalyzer(object):

    def __init__(self, dir_path=None):
        if dir_path is None:
            dir_path = log_dir_path
        self.log_dir_path = dir_path

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
        df = df[df.t >= START_TIME + skip_minutes * 60]
        return df

    def load_vehicle_log(self, max_num=1, skip_minutes=0):
        return self.load_log(self.log_dir_path + vehicle_log_file, vehicle_log_cols, max_num, skip_minutes)

    def load_customer_log(self, max_num=1, skip_minutes=0):
        return self.load_log(self.log_dir_path + customer_log_file, customer_log_cols, max_num, skip_minutes)

    def load_summary_log(self, max_num=1, skip_minutes=0):
        return self.load_log(self.log_dir_path + summary_log_file, summary_log_cols, max_num, skip_minutes)

    def load_score_log(self, max_num=1, skip_minutes=0):
        return self.load_log(self.log_dir_path + score_log_file, score_log_cols, max_num, skip_minutes)

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
        return df

    def add_time_bin(self, df, bin_width):
        return ((df.t - START_TIME) / bin_width).astype(int) * int(bin_width) + START_TIME

