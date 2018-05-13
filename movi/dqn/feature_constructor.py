import numpy as np
from common.time_utils import get_local_datetime
from config.settings import MAP_WIDTH, MAP_HEIGHT
from .settings import FEATURE_MAP_SIZE
from common import vehicle_status_codes, mesh

class FeatureConstructor(object):

    def __init__(self):
        self.t = 0
        self.fingerprint = (100000, 0)

    def update_time(self, current_time):
        self.t = current_time

    def update_supply(self, vehicles, duration=1800):
        idle = vehicles[vehicles.status == vehicle_status_codes.IDLE]
        cruise = vehicles[vehicles.status == vehicle_status_codes.CRUISING]
        occupied = vehicles[vehicles.status == vehicle_status_codes.OCCUPIED]
        occupied = occupied[occupied.time_to_destination <= duration]

        stopped_vehicle_map = self.construct_supply_map(idle[["lon", "lat"]].values)
        cruise_origin_map = self.construct_supply_map(cruise[["lon", "lat"]].values)
        cruise_destination_map = self.construct_supply_map(cruise[["destination_lon", "destination_lat"]].values)
        dropoff_map = self.construct_supply_map(occupied[["destination_lon", "destination_lat"]].values)
        average_map = self.compute_spatial_average(stopped_vehicle_map + cruise_destination_map + dropoff_map)
        self.supply_maps = [stopped_vehicle_map, cruise_origin_map, cruise_destination_map, dropoff_map, average_map]


    def update_demand(self, demand):
        self.demand_maps = demand
        averaged_demand_map = self.compute_spatial_average(self.demand_maps[-1])
        self.demand_maps.append(averaged_demand_map)


    def compute_spatial_average(self, map, radius=10):
        padded_map = np.pad(map, radius, "constant")
        averaged_map = sum([padded_map[x + radius : x + radius + MAP_WIDTH, y + radius : y + radius + MAP_HEIGHT]
                        * np.exp(-(x ** 2  + y ** 2) / (radius / 2.0) ** 2)
                            for x in range(-radius, radius + 1)
                            for y in range(-radius, radius + 1)
                            if x ** 2 + y ** 2 <= radius ** 2]) / ((radius / 2.0) ** 2 * np.pi)
        return averaged_map

    def update_fingerprint(self, fingerprint):
        self.fingerprint = fingerprint

    def construct_features(self, vehicle):
        location = vehicle.lat, vehicle.lon
        features = self.construct_feature_maps(self.get_supply_demand_maps(), location)
        return features, None

    def construct_feature_maps(self, maps, location):
        lat, lon = location
        x, y = mesh.convert_lonlat_to_xy(lon, lat)
        features = [self.extract_box(m, x, y, FEATURE_MAP_SIZE) for m in maps]
        point_map = self.construct_initial_map(w=FEATURE_MAP_SIZE, h=FEATURE_MAP_SIZE)
        center = int((FEATURE_MAP_SIZE - 1) / 2)
        point_map[center, center] = 1.0
        features += [point_map]
        return features

    def get_supply_demand_maps(self):
        supply_demand_maps = self.supply_maps + self.demand_maps
        return supply_demand_maps

    def extract_box(self, F, x, y, size):
        X = self.construct_initial_map(w=size, h=size)
        d = int((size - 1) / 2)
        w, h = F.shape
        X[max(d-x, 0):min(d+w-x, size), max(d-y, 0):min(d+h-y, size)] = F[max(x-d, 0):min(x+d+1, w), max(y-d, 0):min(y+d+1, h)]
        return X


    def construct_initial_map(self, w=MAP_WIDTH, h=MAP_HEIGHT):
        return np.zeros((w, h), dtype=np.float32)


    def construct_supply_map(self, locations):
        supply_map = self.construct_initial_map()
        for lon, lat in locations:
            x, y = mesh.convert_lonlat_to_xy(lon, lat)
            supply_map[x, y] += 1.0
        return supply_map

    # def construct_vehicle_features(self, vehicle):
    #     norm_x, norm_y = self.normalize_lonlat(vehicle.longitude, vehicle.latitude)
    #     return [norm_x, norm_y]

    def construct_time_features(self, timestamp):
        t = get_local_datetime(timestamp)
        hourofday = t.hour / 24.0 * 2 * np.pi
        dayofweek = t.weekday() / 7.0 * 2 * np.pi
        return [np.sin(hourofday), np.cos(hourofday), np.sin(dayofweek), np.cos(dayofweek)]

    def construct_fingerprint_features(self, fingerprint):
        iteration, epsilon = fingerprint
        return [np.log(1 + iteration / 60.0), epsilon]


    def get_current_time(self):
        t = self.t
        return t

    # def normalize_lonlat(self, lon, lat):
    #     x = (lon - CENTER_LONGITUDE) / LON_WIDTH
    #     y = (lat - CENTER_LATITUDE) / LAT_WIDTH
    #     return x, y

