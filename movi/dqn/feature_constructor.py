import numpy as np
from common.time_utils import get_local_datetime
from config.settings import MAP_WIDTH, MAP_HEIGHT
from .settings import MAX_MOVE, NUM_SUPPLY_DEMAND_MAPS
from common import vehicle_status_codes, mesh

class FeatureConstructor(object):

    def __init__(self):
        self.t = 0
        self.fingerprint = (100000, 0)
        self.action_space = [(0, 0)] + [(ax, ay) for ax in range(-MAX_MOVE, MAX_MOVE + 1)
                                        for ay in range(-MAX_MOVE, MAX_MOVE + 1)
                                        if ax ** 2 + ay ** 2 >= 1]
        n = MAX_MOVE * 2 + 1
        self.tt_map = np.zeros((MAP_WIDTH, MAP_HEIGHT, n, n))
        for ax, ay in self.action_space:
            x, y = MAX_MOVE + ax, MAX_MOVE + ay
            self.tt_map[:, :, x, y] = np.sqrt(ax ** 2 + ay ** 2) / (MAX_MOVE + 1)

        self.D_out = np.exp(-self.tt_map)
        for x in range(MAP_WIDTH):
            for y in range(MAP_HEIGHT):
                self.D_out[x, y] /= self.D_out[x, y].sum()

        self.D_in = np.zeros((MAP_WIDTH, MAP_HEIGHT, n, n))
        for x in range(MAP_WIDTH):
            for y in range(MAP_HEIGHT):
                for ax in range(n):
                    for ay in range(n):
                        if x + ax < MAP_WIDTH and y + ay < MAP_HEIGHT:
                            self.D_in[x, y, ax, ay] = self.D_out[x + ax, y + ay, -ax, -ay]


    def update_time(self, current_time):
        self.t = current_time

    def update_supply(self, vehicles, duration=900):
        idle = vehicles[(vehicles.status == vehicle_status_codes.IDLE) | (vehicles.status == vehicle_status_codes.CRUISING)]
        occupied = vehicles[vehicles.status == vehicle_status_codes.OCCUPIED]
        occupied = occupied[occupied.time_to_destination <= duration]
        stopped_vehicle_map = self.construct_supply_map(idle[["lon", "lat"]].values)
        dropoff_map = self.construct_supply_map(occupied[["destination_lon", "destination_lat"]].values)
        self.supply_maps = [stopped_vehicle_map, dropoff_map]

        diffused_smap1 = [self.diffuse_map(sum(self.supply_maps), self.D_in)]
        diffused_smap2 = [self.diffuse_map(s, self.D_in) for s in diffused_smap1]
        diffused_smap3 = [self.diffuse_map(s, self.D_in) for s in diffused_smap2]
        self.diffused_supply = diffused_smap1 + diffused_smap2 + diffused_smap3

    def update_demand(self, demand, normalized_factor=0.1):
        self.demand_maps = [d * normalized_factor for d in demand]

        diffused_dmap1 = [self.diffuse_map(sum(self.demand_maps), self.D_out)]
        diffused_dmap2 = [self.diffuse_map(d, self.D_out) for d in diffused_dmap1]
        diffused_dmap3 = [self.diffuse_map(d, self.D_out) for d in diffused_dmap2]
        self.diffused_demand = diffused_dmap1 + diffused_dmap2 + diffused_dmap3


    def diffuse_map(self, map, d_filter):
        padded_map = np.pad(map, MAX_MOVE, "constant")
        diffused_map = self.construct_initial_map()
        d = MAX_MOVE * 2 + 1
        for x in range(MAP_WIDTH):
            for y in range(MAP_HEIGHT):
                diffused_map[x, y] = (padded_map[x : x + d, y : y + d] * d_filter[x, y]).sum()
        # diffused_map = sum([padded_map[x + MAX_MOVE : x + MAX_MOVE + MAP_WIDTH, y + MAX_MOVE : y + MAX_MOVE + MAP_HEIGHT]
        #                 * np.exp(-(x ** 2  + y ** 2) / (MAX_MOVE ** 2))
        #                     for x, y in self.action_space
        #                     if x ** 2 + y ** 2 <= MAX_MOVE ** 2]) / (MAX_MOVE ** 2 * np.pi)

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
        state_feature = self.construct_state_feature(t, f, l, M)
        actions, action_features = self.construct_action_features(t, l, M)
        s = (state_feature, action_features)
        return s, actions

    def construct_state_feature(self, t, f, l, M):
        x, y = l
        state_feature = [m.mean() for m in M[:NUM_SUPPLY_DEMAND_MAPS]]
        state_feature += [m[x, y] for m in M]
        state_feature += self.construct_location_features(l) + self.construct_time_features(t) + self.construct_fingerprint_features(f)
        return state_feature

    def construct_action_features(self, t, l, M):
        actions = []
        action_features = []

        for ax, ay in self.action_space:
            a = (ax, ay)
            feature = self.construct_action_feature(t, l, M, a)
            if feature is not None:
                actions.append(a)
                action_features.append(feature)

        # rest action
        # rest = 1
        # a = (0, 0, rest)
        # feature = self.construct_action_feature(t, l, M, a)
        # if feature is not None:
        #     actions.append(a)
        #     action_features.append(feature)

        return actions, action_features

    def construct_action_feature(self, t, l, M, a):
        x, y = l
        ax, ay = a
        x_ = x + ax
        y_ = y + ay
        if x_ < MAP_WIDTH and y_ < MAP_HEIGHT and x_ >= 0 and y_ >= 0:
            tt = self.get_triptime(x, y, ax, ay)
            if tt <= 1:
                action_feature = [m[x_, y_] for m in M]
                action_feature += self.construct_location_features((x_, y_)) + [tt]
                return action_feature

        return None


    def get_triptime(self, x, y, ax, ay):
        return self.tt_map[x, y, ax + MAX_MOVE, ay + MAX_MOVE]


    def get_supply_demand_maps(self):
        supply_demand_maps = self.supply_maps + self.demand_maps
        diffused_maps = self.diffused_supply + self.diffused_demand
        return supply_demand_maps + diffused_maps

    # def extract_box(self, F, x, y, size):
    #     X = self.construct_initial_map(w=size, h=size)
    #     d = int((size - 1) / 2)
    #     w, h = F.shape
    #     X[max(d-x, 0):min(d+w-x, size), max(d-y, 0):min(d+h-y, size)] = F[max(x-d, 0):min(x+d+1, w), max(y-d, 0):min(y+d+1, h)]
    #     return X


    def construct_initial_map(self, w=MAP_WIDTH, h=MAP_HEIGHT):
        return np.zeros((w, h), dtype=np.float32)


    # def construct_point_map_feature(self, x, y):
    #     M = self.construct_initial_map(w=FEATURE_MAP_SIZE, h=FEATURE_MAP_SIZE)
    #     M[x, y] = 1.0
    #     return M

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

