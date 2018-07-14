# from config.settings import MAP_WIDTH, MAP_HEIGHT
from common import vehicle_status_codes
from config.settings import DEFAULT_LOG_DIR
import os
import tensorflow as tf


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('offduty_threshold', -float('inf'), 'q value off duty threshold')
flags.DEFINE_float('offduty_probability', 0.20, 'probability to automatically become off duty')
flags.DEFINE_float('alpha', 0.0, 'entropy coefficient')
flags.DEFINE_string('save_memory_dir', os.path.join(DEFAULT_LOG_DIR, 'memory'), 'replay memory storage')
flags.DEFINE_string('save_network_dir', os.path.join(DEFAULT_LOG_DIR, 'networks'), 'network model directory')
flags.DEFINE_string('save_summary_dir', os.path.join(DEFAULT_LOG_DIR, 'summary'), 'training summary directory')
flags.DEFINE_string('load_network', '', "load saved dqn network.")
flags.DEFINE_string('load_memory', '', "load saved replay memory.")

flags.DEFINE_boolean('train', False, "run training dqn network.")
flags.DEFINE_boolean('verbose', False, "print log verbosely.")
flags.DEFINE_integer('pretrain', 0, "run N pretraining steps using pickled experience memory.")
flags.DEFINE_integer('vehicles', 10000, "number of vehicles")
flags.DEFINE_integer('start_time', 1462075200 + 3600 * 4, "simulation start datetime (unixtime)")
flags.DEFINE_integer('start_offset', 0, "simulation start datetime offset (days)")
flags.DEFINE_integer('days', 14, "simulation days")
# flags.DEFINE_string('logdir', 'base', "base log directory name")
flags.DEFINE_string('tag', 'base', "tag used to identify logs")
flags.DEFINE_boolean('log_vehicle', False, "whether to log vehicle states")

# data_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/dqn")
# SAVE_NETWORK_PATH = os.path.join(data_dir_path, 'networks')
# SAVE_SUMMARY_PATH = os.path.join(data_dir_path, 'summary')
# REPLAY_MEMORY_PATH = os.path.join(data_dir_path, 'memory')
# OFFDUTY_THRESHOLD = -1000
# REST_ACTION_PROBABILITY = 0.10
# ALPHA = 0.0 # Entropy Coefficient


STATE_REWARD_TABLE = {
    vehicle_status_codes.IDLE : -0.1,
    vehicle_status_codes.CRUISING : -0.2,
    vehicle_status_codes.ASSIGNED : -0.2,
    vehicle_status_codes.OCCUPIED : -0.2,
    vehicle_status_codes.OFF_DUTY : 0.0
}
EARNINGS_REWARD_FACTOR = 1.0


DEMAND_SUPPLY_UPDATE_CYCLE = 300
GAMMA = 0.99  # Discount Factor
MAX_MOVE = 5
NUM_SUPPLY_DEMAND_MAPS = 4
NUM_FEATURES = 35
WAIT_ACTION_PROBABILITY = 0.70  # wait action probability in epsilon-greedy
EXPLORATION_STEPS = 10000  # Number of steps over which the initial value of epsilon is linearly annealed to its final value
INITIAL_EPSILON = 1.0  # Initial value of epsilon in epsilon-greedy
FINAL_EPSILON = 0.05  # Final value of epsilon in epsilon-greedy
INITIAL_MEMORY_SIZE = 100  # Number of steps to populate the replay memory before training starts
NUM_SUPPLY_DEMAND_HISTORY = 10000
MAX_MEMORY_SIZE = 10000000  # Number of replay memory the agent uses for training
SAVE_INTERVAL = 2000  # The frequency with which the network is saved
BATCH_SIZE = 64  # Mini batch size
NUM_ITERATIONS = 1 # Number of batches
TARGET_UPDATE_INTERVAL = 100  # The frequency with which the target network is updated
LEARNING_RATE = 0.00025  # Learning rate used by RMSProp
MOMENTUM = 0.95  # Momentum used by RMSProp
MIN_GRAD = 0.01  # Constant added to the squared gradient in the denominator of the RMSProp update

