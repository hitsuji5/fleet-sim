import numpy as np
from common.time_utils import get_local_datetime
from config.settings import MAP_WIDTH, MAP_HEIGHT
from .settings import FEATURE_MAP_SIZE, MAX_MOVE, NUM_SUPPLY_DEMAND_MAPS
from common import vehicle_status_codes, mesh

class FeatureConstructor(object):

    def __init__(self):
        self.t = 0
        self.fingerprint = (100000, 0)

    def update_time(self, current_time):
        self.t = current_time

    def update_supply(self, vehicles, duration=900):
        idle = vehicles[(vehicles.status == vehicle_status_codes.IDLE) | (vehicles.status == vehicle_status_codes.CRUISING)]
        occupied = vehicles[vehicles.status == vehicle_status_codes.OCCUPIED]
        occupied = occupied[occupied.time_to_destination <= duration]
        stopped_vehicle_map = self.construct_supply_map(idle[["lon", "lat"]].values)
        dropoff_map = self.construct_supply_map(occupied[["destination_lon", "destination_lat"]].values)
        self.supply_maps = [stopped_vehicle_map, dropoff_map]

        diffused_smap = [self.diffuse_map(stopped_vehicle_map + dropoff_map)]
        diffused_smap += [self.diffuse_map(s) for s in diffused_smap]
        self.diffused_supply = diffused_smap

    def update_demand(self, demand, normalized_factor=0.1):
        self.demand_maps = [d * normalized_factor for d in demand]

        diffused_dmap = [self.diffuse_map(d) for d in self.demand_maps]
        diffused_dmap += [self.diffuse_map(d) for d in diffused_dmap]
        self.diffused_demand = diffused_dmap


    def diffuse_map(self, map, radius=MAX_MOVE*2):
        padded_map = np.pad(map, radius, "constant")
        diffused_map = sum([padded_map[x + radius : x + radius + MAP_WIDTH, y + radius : y + radius + MAP_HEIGHT]
                        * np.exp(-(x ** 2  + y ** 2) / (radius ** 2))
                            for x in range(-radius, radius + 1)
                            for y in range(-radius, radius + 1)
                            if x ** 2 + y ** 2 <= radius ** 2]) / (radius ** 2 * np.pi)

        return diffused_map

    def update_fingerprint(self, fingerprint):
        self.fingerprint = fingerprint

    def construct_current_features(self, x, y):
        M = self.get_supply_demand_maps()
        t = self.get_current_time()
        f = self.get_current_fingerprint()
        l = (x, y)
        s, actions = self.construct_features(t, f, l, M)
        return s, actions

    def construct_features(self, t, f, l, M):
        state_map, state_feature = self.construct_state_feature(t, f, l, M)
        actions, action_maps, action_features = self.construct_action_features(t, l, M)
        s = (state_map, state_feature, action_maps, action_features)
        return s, actions

    def construct_state_feature(self, t, f, l, M):
        x, y = l
        c = int((FEATURE_MAP_SIZE - 1) / 2)
        state_map = [self.extract_box(m, x, y, FEATURE_MAP_SIZE) for m in M[:NUM_SUPPLY_DEMAND_MAPS]]
        state_map.append(self.construct_point_map_feature(c, c))
        state_feature = [m.mean() for m in M[:NUM_SUPPLY_DEMAND_MAPS]]
        state_feature += [m[x, y] for m in M]
        state_feature += self.construct_location_features(l) + self.construct_time_features(t) + self.construct_fingerprint_features(f)
        return state_map, state_feature

    def construct_action_features(self, t, l, M):
        actions = []
        action_maps = []
        action_features = []

        rest = 0
        for ax in range(-MAX_MOVE, MAX_MOVE + 1):
            for ay in range(-MAX_MOVE, MAX_MOVE + 1):
                a = (ax, ay, rest)
                feature = self.construct_action_feature(t, l, M, a)
                if feature is not None:
                    action_map, action_feature = feature
                    actions.append(a)
                    action_maps.append(action_map)
                    action_features.append(action_feature)

        # rest action
        a = (0, 0, 1)
        feature = self.construct_action_feature(t, l, M, a)
        if feature is not None:
            action_map, action_feature = feature
            actions.append(a)
            action_maps.append(action_map)
            action_features.append(action_feature)

        return actions, action_maps, action_features

    def construct_action_feature(self, t, l, M, a):
        x, y = l
        c = int((FEATURE_MAP_SIZE - 1) / 2)
        ax, ay, offduty = a
        x_ = x + ax
        y_ = y + ay
        if x_ < MAP_WIDTH and y_ < MAP_HEIGHT and x_ >= 0 and y_ >= 0:
            tt = self.get_triptime(x, y, x_, y_)
            if tt <= 1:
                action_map = self.construct_point_map_feature(c + ax, c + ay)
                action_feature = [m[x_, y_] for m in M]
                action_feature += self.construct_location_features((x_, y_)) + [tt, offduty]
                return ([action_map], action_feature)

        return None


    def get_triptime(self, sx, sy, tx, ty):
        return np.sqrt((sx - tx) ** 2 + (sy - ty) ** 2) / (MAX_MOVE + 1)


    def get_supply_demand_maps(self):
        supply_demand_maps = self.supply_maps + self.demand_maps
        diffused_maps = self.diffused_demand + self.diffused_supply
        return supply_demand_maps + diffused_maps

    def extract_box(self, F, x, y, size):
        X = self.construct_initial_map(w=size, h=size)
        d = int((size - 1) / 2)
        w, h = F.shape
        X[max(d-x, 0):min(d+w-x, size), max(d-y, 0):min(d+h-y, size)] = F[max(x-d, 0):min(x+d+1, w), max(y-d, 0):min(y+d+1, h)]
        return X


    def construct_initial_map(self, w=MAP_WIDTH, h=MAP_HEIGHT):
        return np.zeros((w, h), dtype=np.float32)


    def construct_point_map_feature(self, x, y):
        M = self.construct_initial_map(w=FEATURE_MAP_SIZE, h=FEATURE_MAP_SIZE)
        M[x, y] = 1.0
        return M

    def construct_supply_map(self, locations):
        supply_map = self.construct_initial_map()
        for lon, lat in locations:
            x, y = mesh.convert_lonlat_to_xy(lon, lat)
            supply_map[x, y] += 1.0
        return supply_map

    def construct_time_features(self, timestamp):
        t = get_local_datetime(timestamp)
        hourofday = t.hour / 24.0 * 2 * np.pi
        dayofweek = t.weekday() / 7.0 * 2 * np.pi
        return [np.sin(hourofday), np.cos(hourofday), np.sin(dayofweek), np.cos(dayofweek)]

    def construct_fingerprint_features(self, fingerprint):
        iteration, epsilon = fingerprint
        return [np.log(1 + iteration / 60.0), epsilon]

    def construct_location_features(self, l):
        x, y = l
        x_norm = float(x - MAP_WIDTH / 2.0) / MAP_WIDTH * 2.0
        y_norm = float(y - MAP_HEIGHT / 2.0) / MAP_HEIGHT * 2.0
        return [x_norm, y_norm]

    def get_current_time(self):
        t = self.t
        return t

    def get_current_fingerprint(self):
        f = self.fingerprint
        return f

