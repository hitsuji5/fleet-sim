import numpy as np
from common import vehicle_status_codes, mesh
from common.geoutils import great_circle_distance
from collections import defaultdict
from config.settings import MAP_WIDTH, MAP_HEIGHT
from simulator.services.routing_service import RoutingEngine

class MatchingPolicy(object):
    def match(self, current_time, vehicles, requests):
        return []

    def find_available_vehicles(self, vehicles):
        idle_vehicles = vehicles[
            ((vehicles.status == vehicle_status_codes.IDLE) |
            (vehicles.status == vehicle_status_codes.CRUISING)) &
            (vehicles.idle_duration > 0)
        ]
        return idle_vehicles

    def create_command(self, vehicle_id, customer_id, duration):
        command = {}
        command["vehicle_id"] = vehicle_id
        command["customer_id"] = customer_id
        command["duration"] = duration
        return command


class RoughMatchingPolicy(MatchingPolicy):
    def __init__(self, reject_distance=5000):
        self.reject_distance = reject_distance # meters

    def match(self, current_time, vehicles, requests):
        assignments = []
        vehicles = self.find_available_vehicles(vehicles)
        n_vehicles = len(vehicles)
        if n_vehicles == 0:
            return assignments
        d = great_circle_distance(vehicles.lat.values, vehicles.lon.values,
                                  requests.origin_lat.values[:, None], requests.origin_lon.values[:, None])
        for rid, customer_id in enumerate(requests.index):
            vid = d[rid].argmin()
            if d[rid, vid] < self.reject_distance:
                vehicle_id = vehicles.index[vid]
                duration = d[rid, vid] / 8.0
                assignments.append(self.create_command(vehicle_id, customer_id, duration))
                d[:, vid] = float('inf')
            else:
                continue
            if len(assignments) == n_vehicles:
                return assignments
        return assignments


class GreedyMatchingPolicy(MatchingPolicy):
    def __init__(self, reject_distance=5000):
        self.reject_distance = reject_distance  # meters
        self.reject_wait_time = 15 * 60         # seconds
        self.k = 3                              # the number of mesh to aggregate
        self.unit_length = 500                  # mesh size in meters
        self.max_locations = 40
        self.routing_engine = RoutingEngine.create_engine()


    def get_coord(self, lon, lat):
        x, y = mesh.convert_lonlat_to_xy(lon, lat)
        return (int(x / self.k), int(y / self.k))

    def coord_iter(self):
        for x in range(int(MAP_WIDTH / self.k)):
            for y in range(int(MAP_HEIGHT / self.k)):
                yield (x, y)


    def find_candidates(self, coord, n_requests, V, reject_range):
        x, y = coord
        candidate_vids = V[(x, y)][:]
        for r in range(1, reject_range):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    r_2 = dx ** 2 + dy ** 2
                    if r ** 2 <= r_2 and r_2 < (r + 1) ** 2:
                        candidate_vids += V[(x + dx, y + dy)][:]
                if len(candidate_vids) > n_requests * 2:
                    break
        return candidate_vids

    def assign_nearest_vehicle(self, request_ids, vehicle_ids, T):
        assignments = []
        for ri, rid in enumerate(request_ids):
            if len(assignments) >= len(vehicle_ids):
                break

            vi = T[ri].argmin()
            tt = T[ri, vi]
            if tt > self.reject_wait_time:
                continue
            vid = vehicle_ids[vi]
            assignments.append((vid, rid, tt))
            T[:, vi] = float('inf')
        return assignments

    def filter_candidates(self, vehicles, requests):
        d = great_circle_distance(vehicles.lat.values, vehicles.lon.values,
                                  requests.origin_lat.mean(), requests.origin_lon.mean())

        within_limit_distance = d < self.reject_distance + self.unit_length * (self.k - 1)
        candidates = vehicles.index[within_limit_distance]
        d = d[within_limit_distance]
        return candidates[np.argsort(d)[:2 * len(requests) + 1]].tolist()

        # while len(d) > 2.0 * len(requests):
        #     r *= decay_rate
        #     d = d[d < r * decay_rate]
        # return vehicles.index[d_ < r].tolist()


    def match(self, current_time, vehicles, requests):
        commands = []
        vehicles = self.find_available_vehicles(vehicles)
        n_vehicles = len(vehicles)
        if n_vehicles == 0:
            return commands

        v_latlon = vehicles[["lat", "lon"]]
        V = defaultdict(list)
        vid2coord = {}
        for vid, row in v_latlon.iterrows():
            coord = self.get_coord(row.lon, row.lat)
            vid2coord[vid] = coord
            V[coord].append(vid)

        r_latlon = requests[["origin_lat", "origin_lon"]]
        R = defaultdict(list)
        for rid, row in r_latlon.iterrows():
            coord = self.get_coord(row.origin_lon, row.origin_lat)
            R[coord].append(rid)

        reject_range = int(self.reject_distance / self.unit_length / self.k) + 1
        for coord in self.coord_iter():
            if not R[coord]:
                continue

            for i in range(int(np.ceil(len(R[coord]) / self.max_locations))):
                target_rids = R[coord][i * self.max_locations : (i + 1) * self.max_locations]

                candidate_vids = self.find_candidates(coord, len(target_rids), V, reject_range)
                if len(candidate_vids) == 0:
                    continue

                target_latlon = r_latlon.loc[target_rids]
                candidate_vids = self.filter_candidates(v_latlon.loc[candidate_vids], target_latlon)
                if len(candidate_vids) == 0:
                    continue
                candidate_latlon = v_latlon.loc[candidate_vids]
                T = self.eta_matrix(candidate_latlon, target_latlon)
                assignments = self.assign_nearest_vehicle(target_rids, candidate_vids, T.T)
                for vid, rid, tt in assignments:
                    commands.append(self.create_command(vid, rid, tt))
                    V[vid2coord[vid]].remove(vid)

        return commands


    def eta_matrix(self, origins_array, destins_array):
        destins = [(lat, lon) for lat, lon in destins_array.values]
        origins = [(lat, lon) for lat, lon in origins_array.values]
        origin_set = list(set(origins))
        latlon2oi = {latlon: oi for oi, latlon in enumerate(origin_set)}
        T = np.array(self.routing_engine.eta_many_to_many([(origin_set, destins)])[0], dtype=np.float32)
        T[np.isnan(T)] = float('inf')
        T = T[[latlon2oi[latlon] for latlon in origins]]
        return T