import numpy as np
from common.time_utils import get_local_datetime
from config.settings import MAP_WIDTH, MAP_HEIGHT
from .settings import FEATURE_MAP_SIZE, MAX_MOVE
from common import vehicle_status_codes, mesh

class FeatureConstructor(object):

    def __init__(self):
        self.t = 0
        self.fingerprint = (100000, 0)

    def update_time(self, current_time):
        self.t = current_time

    def update_supply(self, vehicles, duration=900):
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


    def update_demand(self, demand, normalized_factor=0.1):
        self.demand_maps = [d * normalized_factor for d in demand]
        averaged_dmap = self.compute_spatial_average(self.demand_maps[-1])
        averaged_dmap2 = self.compute_spatial_average(averaged_dmap)

        self.demand_maps += [averaged_dmap, averaged_dmap2]


    def compute_spatial_average(self, map, radius=MAX_MOVE):
        padded_map = np.pad(map, radius, "constant")
        averaged_map = sum([padded_map[x + radius : x + radius + MAP_WIDTH, y + radius : y + radius + MAP_HEIGHT]
                        * np.exp(-(x ** 2  + y ** 2) / (radius / 2.0) ** 2)
                            for x in range(-radius, radius + 1)
                            for y in range(-radius, radius + 1)
                            if x ** 2 + y ** 2 <= radius ** 2]) / ((radius / 2.0) ** 2 * np.pi)

        # smap = self.construct_initial_map()
        # for x in range(MAP_WIDTH):
        #     for y in range(MAP_HEIGHT):
        #         p = np.exp(-self.get_triptime_map(x, y))
        #         smap[x, y] = (p / p.sum() * self.extract_box(map, x, y, MAX_MOVE * 2 + 1)).sum()

        return averaged_map

    def update_fingerprint(self, fingerprint):
        self.fingerprint = fingerprint

    def construct_current_features(self, x, y):
        s, actions = self.construct_features(self.get_supply_demand_maps(), (x, y))
        return s, actions
        # state_map, state_feature = self.construct_state_feature(self.get_supply_demand_maps(), (x, y))
        # actions, action_maps, action_features = self.construct_action_features((x, y))
        # return (state_map, state_feature, action_maps, action_features), actions

    def construct_features(self, maps, location):
        state_map, state_feature = self.construct_state_feature(self.get_supply_demand_maps(), location)
        actions, action_maps, action_features = self.construct_action_features(location)
        return (state_map, state_feature, action_maps, action_features), actions

    def construct_state_feature(self, maps, location):
        x, y = location
        state_map = [self.extract_box(m, x, y, FEATURE_MAP_SIZE) for m in maps]
        point_map = self.construct_initial_map(w=FEATURE_MAP_SIZE, h=FEATURE_MAP_SIZE)
        center = int((FEATURE_MAP_SIZE - 1) / 2)
        point_map[center, center] = 1.0
        state_map += [point_map]
        x_norm, y_norm = self.normalize_xy(x, y)
        state_feature = [x_norm, y_norm] + self.construct_time_features(self.t) + self.construct_fingerprint_features(self.fingerprint)
        return state_map, state_feature

    def construct_action_features(self, location):
        actions = []
        action_features = []
        action_maps = []

        rest = 0
        for ax in range(-MAX_MOVE, MAX_MOVE + 1):
            for ay in range(-MAX_MOVE, MAX_MOVE + 1):
                a = (ax, ay, rest)
                feature = self.construct_action_feature(location, a)
                if feature is not None:
                    action_feature, action_map = feature
                    actions.append(a)
                    action_features.append(action_feature)
                    action_maps.append(action_map)

        # rest action
        a = (0, 0, 1)
        action_feature, action_map = self.construct_action_feature(location, a)
        actions.append(a)
        action_features.append(action_feature)
        action_maps.append(action_map)

        return actions, action_maps, action_features

    def construct_action_feature(self, loc, a):
        x, y = loc
        c = int((FEATURE_MAP_SIZE - 1) / 2)
        ax, ay, offduty = a
        x_ = x + ax
        y_ = y + ay
        if x_ < MAP_WIDTH and y_ < MAP_HEIGHT and x_ >= 0 and y_ >= 0:
            tt = self.get_triptime(x, y, x_, y_) / (MAX_MOVE + 1)
            if tt <= 1:
                x_norm, y_norm = self.normalize_xy(x_, y_)
                action_feature = [x_norm, y_norm, tt, offduty]
                action_map = self.construct_initial_map(w=FEATURE_MAP_SIZE, h=FEATURE_MAP_SIZE)
                action_map[c + ax, c + ay] = 1.0
                return (action_feature, action_map)

        return None


    def get_triptime(self, sx, sy, tx, ty):
        return np.sqrt((sx - tx) ** 2 + (sy - ty) ** 2)


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

    def normalize_xy(self, x, y):
        x_norm = float(x - MAP_WIDTH / 2.0) / MAP_WIDTH * 2.0
        y_norm = float(y - MAP_HEIGHT / 2.0) / MAP_HEIGHT * 2.0
        return x_norm, y_norm

