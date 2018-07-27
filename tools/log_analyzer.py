import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../movi/')
from dqn.settings import WORKING_COST, DRIVING_COST
from common import time_utils, vehicle_status_codes, customer_status_codes
from config import settings

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
    "n_dispatch",
    "average_wt"
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
        total_seconds = (df.t.max() - df.t.min() + 3600 * 24)
        df = df[df.t == df.t.max()]
        df["working"] = total_seconds - df.offduty
        df["occupancy_rate"] = df.occupied / df.working
        df["cruising_rate"] = (df.cruising + df.assigned) / df.working
        df["working_rate"] = df.working / (df.working + df.offduty)
        df["reward"] = df.earning \
                       - (df.cruising + df.assigned + df.occupied) * DRIVING_COST / settings.TIMESTEP \
                       - df.working * WORKING_COST / settings.TIMESTEP
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


    def plot_summary(self, path, plt):
        summary = self.load_summary_log(path)
        plt.figure(figsize=(12, 10))
        plt.subplot(311)
        plt.plot(summary.t, summary.n_requests, label="request", alpha=0.7)
        plt.plot(summary.t, summary.n_requests - summary.n_matching, label="reject", alpha=0.7)
        plt.plot(summary.t, summary.n_dispatch, label="dispatch", alpha=0.7)
        plt.ylim([0, 700])
        plt.legend()
        plt.subplot(312)
        plt.plot(summary.t, summary.n_vehicles, label="n_vehicles")
        plt.plot(summary.t, summary.occupied_vehicles, label="n_occupied")
        plt.legend()

        plt.subplot(313)
        plt.plot(summary.t, summary.average_wt, alpha=1.0)
        return plt

    def plot_metrics(self, paths, labels, n_days, plt):
        data = []
        plt.figure(figsize=(12, 7))
        for p, label in zip(paths, labels):
            score = self.load_score_log(p)
            c = self.load_customer_log(p)

            plt.subplot(231)
            plt.title("revenue/hour")
            plt.hist(score.earning / score.working * 3600, bins=100, range=(10, 50), alpha=0.3, label=label)
            plt.yticks([])

            plt.subplot(232)
            plt.title("working rate")
            plt.hist(score.working_rate, bins=100, range=(0, 1), alpha=0.3, label=label)
            plt.yticks([])

            plt.subplot(233)
            plt.title("cruising rate")
            plt.hist(score.cruising_rate, bins=100, range=(0, 1), alpha=0.3, label=label)
            plt.yticks([])

            plt.subplot(234)
            plt.title("occupancy rate")
            plt.hist(score.occupancy_rate, bins=100, range=(0, 1), alpha=0.3, label=label)
            plt.yticks([])

            plt.subplot(235)
            plt.title("total reward / day")
            plt.hist(score.reward / n_days, bins=100, range=(-100, 500), alpha=0.3, label=label)
            plt.yticks([])

            plt.subplot(236)
            plt.title("waiting time")
            plt.hist(c[c.status==2].waiting_time, bins=500, range=(0, 650), alpha=0.3, label=label)
            plt.yticks([])

            x = {}
            x["0_reject_rate"] = float(len(c[c.status == 4])) / len(c)
            x["1_occupancy_rate"] = (score.occupied / score.working).mean()
            x["2_revenue/hour"] = float(score.earning.sum()) / score.working.sum() * 3600
            x["3_working/day"] = float(score.working.sum()) / n_days / len(score) / 3600
            x["4_cruising/day"] = (score.cruising + score.assigned).sum() / n_days / len(score) / (
            3600)
            x["5_waiting_time"] = c[c.status == 2].waiting_time.mean()
            data.append(x)

        plt.legend()
        df = pd.DataFrame(data, index=labels)
        return plt, df
