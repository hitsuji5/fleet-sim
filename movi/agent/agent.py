from common import vehicle_status_codes

class Agent(object):

    def __init__(self, dispatch_policy, matching_policy):
        self.matching_policy = matching_policy
        self.dispatch_policy = dispatch_policy

    def get_commands(self, current_time, vehicles, requests):
        matching_commands = []
        if len(requests) > 0:
            matching_commands = self.matching_policy.match(current_time, vehicles, requests)
            vehicles = self.update_vehicles(vehicles, matching_commands)

        dispatch_commands = self.dispatch_policy.dispatch(current_time, vehicles)
        return matching_commands, dispatch_commands


    def update_vehicles(self, vehicles, commands):
        vehicle_ids = [command["vehicle_id"] for command in commands]
        vehicles.loc[vehicle_ids, "status"] = vehicle_status_codes.ASSIGNED
        return vehicles



