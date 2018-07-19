from common import vehicle_status_codes
from .services.demand_prediction_service import DemandPredictionService
from config.settings import TIMESTEP, MIN_DISPATCH_CYCLE, MAX_DISPATCH_CYCLE
import numpy as np

class DispatchPolicy(object):
    def __init__(self):
        self.demand_predictor = DemandPredictionService()
        self.updated_at = {}

    def dispatch(self, current_time, vehicles):
        tbd_vehicles = self.get_tbd_vehicles(vehicles, current_time)
        if len(tbd_vehicles) == 0:
            return []

        self.update_state(current_time, vehicles)
        commands = self.get_commands(tbd_vehicles)
        self.record_DISPATCHd_at(tbd_vehicles.index, current_time)

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
            if current_time - self.updated_at.get(vehicle_id, 0) >= MIN_DISPATCH_CYCLE
        ]]
        tbd_cruising_vehicles = cruising_vehicles.loc[[
            vehicle_id for vehicle_id in cruising_vehicles.index
            if current_time - self.updated_at.get(vehicle_id, 0) >= MAX_DISPATCH_CYCLE
        ]]

        tbd_vehicles = tbd_idle_vehicles.append(tbd_cruising_vehicles)
        max_n = int(len(vehicles) / MIN_DISPATCH_CYCLE * TIMESTEP)
        if len(tbd_vehicles) > max_n:
            p = np.random.permutation(len(tbd_vehicles))
            tbd_vehicles = tbd_vehicles.iloc[p[:max_n]]
        return tbd_vehicles

    def record_DISPATCHd_at(self, vehicle_ids, current_time):
        for vehicle_id in vehicle_ids:
            self.updated_at[vehicle_id] = current_time

    def create_command(self, vehicle_id, destination=None, offduty=False, cache_key=None):
        command = {}
        command["vehicle_id"] = vehicle_id
        if offduty:
            command["offduty"] = True
        elif cache_key is not None:
            command["cache"] = cache_key
        else:
            command["destination"] = destination
        return command