import numpy as np
from simulator.simulator import Simulator
from agent.agent import Agent
from common import vehicle_status_codes, mesh
from config.settings import MAP_WIDTH, MAP_HEIGHT
from logger import sim_logger

class Experiment(object):

    def __init__(self, start_time, timestep, dispatch_policy, matching_policy):
        self.simulator = Simulator(start_time, timestep)
        self.agent = Agent(dispatch_policy, matching_policy)

    def reset(self, start_time=None, timestep=None):
        self.simulator.log_score()
        self.simulator.reset(start_time, timestep)

    def populate_vehicles(self, n_vehicles):
        vehicle_ids = range(1, n_vehicles + 1)
        locations = [mesh.convert_xy_to_lonlat(x, y)[::-1] for x in range(MAP_WIDTH) for y in range(MAP_HEIGHT)]
        p = self.agent.dispatch_policy.demand_predictor.predict(self.simulator.get_current_time())[0]
        p = p.flatten() / p.sum()
        vehicle_locations = [locations[i] for i in np.random.choice(len(locations), size=n_vehicles, p=p)]
        self.simulator.populate_vehicles(vehicle_ids, vehicle_locations)


    def step(self, verbose=False):
        self.simulator.step()
        vehicles = self.simulator.get_vehicles_state()
        requests = self.simulator.get_new_requests()
        current_time = self.simulator.get_current_time()
        m_commands, d_commands = self.agent.get_commands(current_time, vehicles, requests)
        self.simulator.match_vehicles(m_commands)
        self.simulator.dispatch_vehicles(d_commands)

        net_v = vehicles[vehicles.status != vehicle_status_codes.OFF_DUTY]
        average_wt = np.mean([command['duration'] for command in m_commands]).astype(int)
        summary = "{:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}".format(
            current_time, len(net_v), len(net_v[net_v.status == vehicle_status_codes.OCCUPIED]),
            len(requests), len(m_commands), len(d_commands), average_wt
        )
        sim_logger.log_summary(summary)

        if verbose:
            print("summary: ({})".format(summary), flush=True)

    def dry_run(self, n_steps):
        for _ in range(n_steps):
            self.simulator.step()