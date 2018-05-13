import numpy as np
from simulator.simulator import Simulator
from agent.agent import Agent
from common import vehicle_status_codes, mesh
from config.settings import MAP_WIDTH, MAP_HEIGHT

class Experiment(object):

    def __init__(self, start_time, timestep, dispatch_policy, matching_policy, use_pattern=False):
        self.simulator = Simulator(start_time, timestep, use_pattern)
        self.agent = Agent(dispatch_policy, matching_policy)


    def init_vehicles(self, n_vehicles, n_steps=10):
        vehicle_ids = range(1, n_vehicles + 1)
        N = int(n_vehicles / n_steps)
        locations = [mesh.convert_xy_to_lonlat(x ,y)[::-1] for x in range(MAP_WIDTH) for y in range(MAP_HEIGHT)]
        for i in range(n_steps):
            vehicle_locations = [locations[np.random.randint(0, len(locations))] for _ in range(N)]
            self.simulator.populate_vehicles(vehicle_ids[i * N : (i + 1) * N], vehicle_locations)
            self.simulator.step()
            requests = self.simulator.get_new_requests()
            vehicles = self.simulator.get_vehicles_state()
            current_time = self.simulator.get_current_time()
            m_commands, d_commands = self.agent.get_commands(current_time, vehicles, requests)
            self.simulator.match_vehicles(m_commands)
            self.simulator.dispatch_vehicles(d_commands)

    def step(self, verbose=False):
        self.simulator.step()
        vehicles = self.simulator.get_vehicles_state()
        requests = self.simulator.get_new_requests()
        current_time = self.simulator.get_current_time()
        # commands = self.agent.get_commands(current_time, vehicles, requests)
        # self.simulator.dispatch_vehicles(commands)
        m_commands, d_commands = self.agent.get_commands(current_time, vehicles, requests)
        self.simulator.match_vehicles(m_commands)
        self.simulator.dispatch_vehicles(d_commands)

        if verbose:
            print("t={:d}, n_vehicles={:d}, n_requests={:d}, matching={:d}, dispatch={:d}".format(
                current_time, len(vehicles[vehicles.status != vehicle_status_codes.OFF_DUTY]),
                len(requests), len(m_commands), len(d_commands)))

    def dry_run(self, n_steps):
        for _ in range(n_steps):
            self.simulator.step()