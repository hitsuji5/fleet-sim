# from config.settings import MAP_WIDTH, MAP_HEIGHT
from common import vehicle_status_codes
import os

MAX_MOVE = 4
FEATURE_MAP_SIZE = (MAX_MOVE + 6) * 2 + 1
INPUT_SHAPE = (FEATURE_MAP_SIZE, FEATURE_MAP_SIZE, 9, )
WAIT_ACTION = int(((MAX_MOVE * 2 + 1) ** 2 - 1) / 2)

EARNINGS_REWARD_FACTOR = 0.1
STATE_REWARD_TABLE = {
    vehicle_status_codes.IDLE : 0.0,
    vehicle_status_codes.CRUISING : -0.02,
    vehicle_status_codes.ASSIGNED : -0.02,
    vehicle_status_codes.OCCUPIED : 0.0,
    vehicle_status_codes.OFF_DUTY : 0.0
}

GAMMA = 0.99
EXPLORATION_STEPS = 5000  # Number of steps over which the initial value of epsilon is linearly annealed to its final value
INITIAL_EPSILON = 1.0  # Initial value of epsilon in epsilon-greedy
FINAL_EPSILON = 0.05  # Final value of epsilon in epsilon-greedy
WAIT_ACTION_PROBABILITY = 0.7
INITIAL_MEMORY_SIZE = 500  # Number of steps to populate the replay memory before training starts
NUM_SUPPLY_DEMAND_HISTORY = 10000
MAX_MEMORY_SIZE = 1500000  # Number of replay memory the agent uses for training
SAVE_INTERVAL = 1000  # The frequency with which the network is saved
BATCH_SIZE = 32  # Mini batch size
NUM_ITERATIONS = 4 # Number of batches
TARGET_UPDATE_INTERVAL = 50  # The frequency with which the target network is updated
LEARNING_RATE = 0.00025  # Learning rate used by RMSProp
MOMENTUM = 0.95  # Momentum used by RMSProp
MIN_GRAD = 0.01  # Constant added to the squared gradient in the denominator of the RMSProp update

data_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/dqn")
SAVE_NETWORK_PATH = os.path.join(data_dir_path, 'network')
SAVE_SUMMARY_PATH = os.path.join(data_dir_path, 'summary')
REPLAY_MEMORY_PATH = os.path.join(data_dir_path, 'memory')