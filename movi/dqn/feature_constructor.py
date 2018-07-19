import numpy as np
import os
from common.time_utils import get_local_datetime
from config.settings import MAP_WIDTH, MAP_HEIGHT, DATA_DIR, MIN_DISPATCH_CYCLE
from .settings import MAX_MOVE, NUM_SUPPLY_DEMAND_MAPS, FLAGS
from common import vehicle_status_codes, mesh


class FeatureConstructor(object):

    def __init__(self):
        self.t = 0
        self.fingerprint = (100000, 0)
        # self.action_space = [(0, 0)] + [(ax, ay) for ax in range(-MAX_MOVE, MAX_MOVE + 1)
        #                                 for ay in range(-MAX_MOVE, MAX_MOVE + 1)
        #                                 if ax ** 2 + ay ** 2 >= 1]
        self.reachable_map = self.load_reachable_map()
        self.state_space = [(x, y) for x in range(MAP_WIDTH) for y in range(MAP_HEIGHT) if self.reachable_map[x, y] == 1]
        self.tt_map = self.load_tt_map()
        self.D_out, self.D_in = self.build_diffusion_filter()
        self.d_entropy = self.build_diffusion_entropy_map()


    def action_space_iter(self, x, y):
        yield (0, 0)
        for ax in range(-MAX_MOVE, MAX_MOVE + 1):
            for ay in range(-MAX_MOVE, MAX_MOVE + 1):
                if ax == 0 and ay == 0:
                    continue
                x_ = x + ax
                y_ = y + ay
                if self.is_reachable(x_, y_):
                    yield (ax, ay)


    def load_reachable_map(self):
        return np.load(os.path.join(DATA_DIR, 'reachable_map.npy'))

        # if FLAGS.use_osrm:
        #     return np.load(os.path.join(DATA_DIR, 'reachable_map.npy'))
        # return np.ones((MAP_WIDTH, MAP_HEIGHT))

    def load_tt_map(self):
        return np.load(os.path.join(DATA_DIR, 'tt_map.npy')) / MIN_DISPATCH_CYCLE / 2.0

        # if FLAGS.use_osrm:
        #     return np.load(os.path.join(DATA_DIR, 'tt_map.npy')) / MAX_DISPATCH_CYCLE
        #
        # n = MAX_MOVE * 2 + 1
        # tt_map = np.ones((MAP_WIDTH, MAP_HEIGHT, n, n)) * float('inf')
        #
        # for x, y in self.state_space:
        #     for ax, ay in self.action_space_iter(x, y):
        #         axi, ayi = MAX_MOVE + ax, MAX_MOVE + ay
        #         tt_map[x, y, axi, ayi] = np.sqrt(ax ** 2 + ay ** 2) / (MAX_MOVE + 1)
        # return tt_map

    def build_diffusion_filter(self):
        D_out = np.exp(-self.tt_map * 2)
        for x, y in self.state_space:
            D_out[x, y] /= D_out[x, y].sum()

        n = MAX_MOVE * 2 + 1
        D_in = np.zeros((MAP_WIDTH, MAP_HEIGHT, n, n))
        for x, y in self.state_space:
            for ax, ay in self.action_space_iter(x, y):
                axi, ayi = MAX_MOVE + ax, MAX_MOVE + ay
                D_in[x, y, axi, ayi] = D_out[x + ax, y + ay, -axi, -ayi]
        return D_out, D_in


    def build_diffusion_entropy_map(self):
        entropy = np.zeros((MAP_WIDTH, MAP_HEIGHT))
        for x, y in self.state_space:
            entropy[x, y] = -(self.D_out[x, y] * np.log(self.D_out[x, y] + 1e-6)).sum()
        entropy /= np.log((MAX_MOVE * 2 + 1) ** 2 + 1e-6)
        diffused_entropy = [entropy] + self.diffusion_convolution(entropy, self.D_out, FLAGS.n_diffusions - 1)
        return diffused_entropy

    def update_time(self, current_time):
        self.t = current_time

    def update_supply(self, vehicles, duration=MIN_DISPATCH_CYCLE * 2):
        idle = vehicles[(vehicles.status == vehicle_status_codes.IDLE) | (vehicles.status == vehicle_status_codes.CRUISING)]
        occupied = vehicles[vehicles.status == vehicle_status_codes.OCCUPIED]
        occupied = occupied[occupied.time_to_destination <= duration]
        stopped_vehicle_map = self.construct_supply_map(idle[["lon", "lat"]].values)
        dropoff_map = self.construct_supply_map(occupied[["destination_lon", "destination_lat"]].values)
        self.supply_maps = [stopped_vehicle_map, dropoff_map]
        self.diffused_supply = self.diffusion_convolution(sum(self.supply_maps), self.D_in, FLAGS.n_diffusions)

    def update_demand(self, demand, normalized_factor=0.1):
        self.demand_maps = [d * normalized_factor for d in demand]
        self.diffused_demand = self.diffusion_convolution(sum(self.demand_maps), self.D_out, FLAGS.n_diffusions)

    def diffusion_convolution(self, map, d_filter, k):
        M = map
        diffused_maps = []
        for _ in range(k):
            M = self.diffuse_map(M, d_filter)
            diffused_maps.append(M)
        return diffused_maps


    def diffuse_map(self, map, d_filter):
        padded_map = np.pad(map, MAX_MOVE, "constant")
        diffused_map = self.construct_initial_map()
        d = MAX_MOVE * 2 + 1
        for x in range(MAP_WIDTH):
            for y in range(MAP_HEIGHT):
                diffused_map[x, y] = (padded_map[x : x + d, y : y + d] * d_filter[x, y]).sum()
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
        state_feature += [m[x, y] for m in self.d_entropy]
        state_feature += self.construct_location_features(l) + self.construct_time_features(t) + self.construct_fingerprint_features(f)
        return state_feature

    def construct_action_features(self, t, l, M):
        actions = []
        action_features = []

        for ax, ay in self.action_space_iter(*l):
            a = (ax, ay)
            feature = self.construct_action_feature(t, l, M, a)
            if feature is not None:
                actions.append(a)
                action_features.append(feature)

        return actions, action_features

    def is_reachable(self, x, y):
        return 0 <= x and x < MAP_WIDTH and 0 <= y and y < MAP_HEIGHT and self.reachable_map[x, y] == 1

    def construct_action_feature(self, t, l, M, a):
        x, y = l
        ax, ay = a
        x_ = x + ax
        y_ = y + ay
        # if self.is_reachable(x_, y_):
        tt = self.get_triptime(x, y, ax, ay)
        if tt <= 1:
            action_feature = [m[x_, y_] for m in M]
            action_feature += [m[x_, y_] for m in self.d_entropy]
            action_feature += self.construct_location_features((x_, y_)) + [tt]
            return action_feature

        return None


    def get_triptime(self, x, y, ax, ay):
        return self.tt_map[x, y, ax + MAX_MOVE, ay + MAX_MOVE]


    def get_supply_demand_maps(self):
        supply_demand_maps = self.supply_maps + self.demand_maps
        diffused_maps = self.diffused_supply + self.diffused_demand
        return supply_demand_maps + diffused_maps

    def construct_initial_map(self, w=MAP_WIDTH, h=MAP_HEIGHT):
        return np.zeros((w, h), dtype=np.float32)

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

