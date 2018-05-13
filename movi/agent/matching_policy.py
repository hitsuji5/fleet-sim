from common import vehicle_status_codes
from common.geoutils import great_circle_distance

class MatchingPolicy(object):
    def match(self, current_time, vehicles, requests):
        return []

    def find_available_vehicles(self, vehicles):
        idle_vehicles = vehicles[(vehicles.status == vehicle_status_codes.IDLE) |
                                 (vehicles.status == vehicle_status_codes.CRUISING)]
        return idle_vehicles

    def create_command(self, vehicle_id, customer_id):
        command = {}
        command["vehicle_id"] = vehicle_id
        command["customer_id"] = customer_id
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
                assignments.append(self.create_command(vehicle_id, customer_id))
                d[:, vid] = float('inf')
            else:
                continue
            if len(assignments) == n_vehicles:
                return assignments
        return assignments
