from config.settings import LON_WIDTH, LAT_WIDTH
from common import vehicle_status_codes

DELTA_LON = 45.0 / 3600
DELTA_LAT = 30.0 / 3600
MAP_WIDTH = int(LON_WIDTH * 2 / DELTA_LON)
MAP_HEIGHT = int(LAT_WIDTH * 2 / DELTA_LAT)
INPUT_SHAPE = (MAP_WIDTH, MAP_HEIGHT, 10, )
MAX_MOVE = 4
FEATURE_MAP_SIZE = (MAX_MOVE + 6) * 2 + 1

EARNINGS_REWARD_FACTOR = 0.1
STATE_REWARD_TABLE = {
    vehicle_status_codes.IDLE : -0.01,
    vehicle_status_codes.CRUISING : -0.02,
    vehicle_status_codes.ASSIGNED : -0.02,
    vehicle_status_codes.OCCUPIED : -0.02,
    vehicle_status_codes.OFF_DUTY : 0.0
}

GAMMA = 0.99
EXPLORATION_STEPS = 5000  # Number of steps over which the initial value of epsilon is linearly annealed to its final value
INITIAL_EPSILON = 1.0  # Initial value of epsilon in epsilon-greedy
FINAL_EPSILON = 0.05  # Final value of epsilon in epsilon-greedy
INITIAL_MEMORY_SIZE = 3000  # Number of steps to populate the replay memory before training starts
NUM_SUPPLY_DEMAND_HISTORY = 10000
MAX_MEMORY_SIZE = 150000  # Number of replay memory the agent uses for training
SAVE_INTERVAL = 1000  # The frequency with which the network is saved
BATCH_SIZE = 32  # Mini batch size
NUM_ITERATIONS = 2 # Number of batches
TARGET_UPDATE_INTERVAL = 100  # The frequency with which the target network is updated
LEARNING_RATE = 0.00025  # Learning rate used by RMSProp
MOMENTUM = 0.95  # Momentum used by RMSProp
MIN_GRAD = 0.01  # Constant added to the squared gradient in the denominator of the RMSProp update
SAVE_NETWORK_PATH = 'data/models/saved_networks/'
SAVE_SUMMARY_PATH = 'data/models/summary/'
REPLAY_MEMORY_PATH = 'data/models/memory/'
