from common import vehicle_status_codes
from .services.demand_prediction_service import DemandPredictionService
from config.settings import DEMAND_AMPLIFICATION_FACTOR


class DispatchPolicy(object):
    def __init__(self, min_update_cycle=300):
        self.min_update_cycle = min_update_cycle # seconds
        self.demand_predictor = DemandPredictionService(amplification_factor=DEMAND_AMPLIFICATION_FACTOR)
        self.updated_at = {}

    def dispatch(self, current_time, vehicles):
        tbd_vehicles = self.get_tbd_vehicles(vehicles, current_time)
        if len(tbd_vehicles) == 0:
            return []

        self.update_state(current_time, vehicles)
        commands = self.get_commands(tbd_vehicles)
        self.record_updated_at(tbd_vehicles.index, current_time)

        return commands

    def update_state(self, current_time, vehicles):
        pass

    def get_commands(self, tbd_vehicles):
        return []

    def get_tbd_vehicles(self, vehicles, current_time):
        idle_vehicles = vehicles[vehicles.status == vehicle_status_codes.IDLE]
        cruising_vehicles = vehicles[vehicles.status == vehicle_status_codes.CRUISING]
        tbd_idle_vehicles = idle_vehicles.loc[[
            vehicle_id for vehicle_id in idle_vehicles.index
            if current_time - self.updated_at.get(vehicle_id, 0) >= self.min_update_cycle / 2
        ]]
        tbd_cruising_vehicles = cruising_vehicles.loc[[
            vehicle_id for vehicle_id in cruising_vehicles.index
            if current_time - self.updated_at.get(vehicle_id, 0) >= self.min_update_cycle
        ]]

        tbd_vehicles = tbd_idle_vehicles.append(tbd_cruising_vehicles)

        return tbd_vehicles

    def record_updated_at(self, vehicle_ids, current_time):
        for vehicle_id in vehicle_ids:
            self.updated_at[vehicle_id] = current_time

    def create_command(self, vehicle_id, destination):
        command = {}
        command["vehicle_id"] = vehicle_id
        command["destination"] = destination
        return command