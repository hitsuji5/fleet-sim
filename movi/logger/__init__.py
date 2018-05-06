import os
import logging.config
from logging import getLogger

import yaml

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logging.yaml')

class SimulationLogger(object):

    def __init__(self, path, level=logging.INFO):
        self.setup_logging(path, level)
        self.env = None
        self.vehicle_logger = getLogger('vehicle')
        self.customer_logger = getLogger('customer')
        self.command_logger = getLogger('command')

    def setup_logging(self, path, level):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)

    def set_env(self, env):
        self.env = env

    def get_current_time(self):
        if self.env:
            return self.env.get_current_time()
        return 0

    def log_vehicle_event(self, event, state):
        t = self.get_current_time()
        self.vehicle_logger.info('{},{},{}'.format(str(t), event, state))

    def log_customer_event(self, event, state):
        t = self.get_current_time()
        self.customer_logger.info('{},{},{}'.format(str(t), event, state))

    # def log_command(self, event, vehicle_id):
    #     pass

sim_logger = SimulationLogger(path)