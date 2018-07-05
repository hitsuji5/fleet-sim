import pickle
import os
import numpy as np
from collections import OrderedDict, defaultdict
from .feature_constructor import FeatureConstructor
from .q_network import DeepQNetwork, FittingDeepQNetwork
from agent.dispatch_policy import DispatchPolicy
from . import settings
from common.time_utils import get_local_datetime
from common import vehicle_status_codes, mesh

class DQNDispatchPolicy(DispatchPolicy):

    def __init__(self):
        super().__init__()
        self.feature_constructor = FeatureConstructor()
        self.q_network = None
        self.q_cache = {}


    def build_q_network(self, load_network=False):
        self.q_network = DeepQNetwork(load_network)

    def update_state(self, current_time, vehicles):
        t = self.feature_constructor.get_current_time()
        if t == 0 or current_time % settings.DEMAND_SUPPLY_UPDATE_CYCLE == 0:
            self.q_cache = {}
            self.feature_constructor.update_supply(vehicles)
            demand = self.demand_predictor.predict(current_time, horizon=2)
            self.feature_constructor.update_demand(demand)

        self.feature_constructor.update_time(current_time)


    def get_commands(self, tbd_vehicles):
        commands = []
        for vehicle_id, vehicle_state in tbd_vehicles.iterrows():
            ax, ay, offduty = self.predict_best_action(vehicle_id, vehicle_state)
            if offduty:
                command = self.create_command(vehicle_id, None, offduty=True)
            else:
                target = self.convert_action_to_destination(vehicle_state, ax, ay)
                if target is None:
                    continue
                command = self.create_command(vehicle_id, target)
            commands.append(command)
        return commands

    def predict_best_action(self, vehicle_id, vehicle_state):
        x, y = mesh.convert_lonlat_to_xy(vehicle_state.lon, vehicle_state.lat)
        if self.q_network is None:
            return settings.WAIT_ACTION

        if (x, y) in self.q_cache:
            actions, p = self.q_cache[(x, y)]
        else:
            s, actions = self.feature_constructor.construct_current_features(x, y)
            q_values = self.q_network.compute_q_values(s)
            p = self.q_network.get_probability(q_values)
            self.q_cache[(x, y)] = actions, p
        aidx = self.q_network.get_action(p)
        return actions[aidx]


    def convert_action_to_destination(self, vehicle_state, ax, ay):
        x, y = mesh.convert_lonlat_to_xy(vehicle_state.lon, vehicle_state.lat)
        lon, lat = mesh.convert_xy_to_lonlat(x + ax, y + ay)
        if lon == vehicle_state.lon and lat == vehicle_state.lat:
            return None
        return lat, lon



class DQNDispatchPolicyLearner(DQNDispatchPolicy):

    def __init__(self):
        super().__init__()
        self.supply_demand_history = OrderedDict()
        self.experience_memory = []
        self.last_state_actions = {}
        self.rewards = defaultdict(int)
        self.last_earnings = defaultdict(int)


    def reset(self):
        self.last_state_actions = {}
        self.rewards = defaultdict(int)
        self.last_earnings = defaultdict(int)

    def dump_experience_memory(self, t):
        sd_path = os.path.join(settings.REPLAY_MEMORY_PATH, "sd_history_{}.pkl".format(t))
        sars_path = os.path.join(settings.REPLAY_MEMORY_PATH, "sars_history_{}.pkl".format(t))
        pickle.dump(self.supply_demand_history, open(sd_path, "wb"))
        pickle.dump(self.experience_memory, open(sars_path, "wb"))

    def load_experience_memory(self, t):
        sd_path = os.path.join(settings.REPLAY_MEMORY_PATH, "sd_history_{}.pkl".format(t))
        sars_path = os.path.join(settings.REPLAY_MEMORY_PATH, "sars_history_{}.pkl".format(t))
        self.supply_demand_history = pickle.load(open(sd_path, "rb"))
        self.experience_memory = pickle.load(open(sars_path, "rb"))

        state_action, _, _ = self.experience_memory[0]
        t_start, _, _, _ = state_action
        state_action, _, _ = self.experience_memory[-1]
        t_end, _, _, _ = state_action
        print("period: {} ~ {}".format(get_local_datetime(t_start), get_local_datetime(t_end)))


    def build_q_network(self, load_network=False):
        self.q_network = FittingDeepQNetwork(load_network)

    def get_commands(self, tbd_vehicles):
        # rewards = [self.rewards[vehicle_id] for vehicle_id in tbd_vehicles.index]
        # print("average_reward : mean {:.2f}, min {:.2f}, max {:.2f} ".format(np.mean(rewards), np.min(rewards), np.max(rewards)))
        commands = super().get_commands(tbd_vehicles)
        return commands

    def predict_best_action(self, vehicle_id, vehicle_state):
        if settings.WAIT_ACTION_PROBABILITY * self.q_network.epsilon > np.random.random():
            if settings.REST_ACTION_PROBABILITY > np.random.random():
                a = settings.REST_ACTION
            else:
                a = settings.WAIT_ACTION
        else:
            a = super().predict_best_action(vehicle_id, vehicle_state)
        self.memorize_experience(vehicle_id, vehicle_state, a)
        return a

    def give_rewards(self, vehicles):
        for vehicle_id, row in vehicles.iterrows():
            earnings = row.earnings - self.last_earnings.get(vehicle_id, 0)
            self.rewards[vehicle_id] += earnings * settings.EARNINGS_REWARD_FACTOR + settings.STATE_REWARD_TABLE[row.status]
            self.last_earnings[vehicle_id] = row.earnings

    def dispatch(self, current_time, vehicles):
        self.give_rewards(vehicles)
        commands = super().dispatch(current_time, vehicles)

        f = self.q_network.get_fingerprint()
        self.feature_constructor.update_fingerprint(f)
        self.backup_supply_demand()

        if len(self.supply_demand_history) > settings.INITIAL_MEMORY_SIZE:
            # if self.q_network.n_steps == 0:
            #     print("Dumping experience memory as pickle...")
            #     self.dump_experience_memory()

            average_loss, average_q_max = self.train_network(settings.BATCH_SIZE, settings.NUM_ITERATIONS)
            print("iterations : {}, average_loss : {:.3f}, average_q_max : {:.3f}".format(self.q_network.n_steps, average_loss, average_q_max))
            self.q_network.write_summary(average_loss, average_q_max)
        return commands


    def backup_supply_demand(self):
        current_time = self.feature_constructor.get_current_time()
        self.supply_demand_history[current_time] = self.feature_constructor.get_supply_demand_maps()

        if len(self.supply_demand_history) > settings.NUM_SUPPLY_DEMAND_HISTORY:
            self.supply_demand_history.popitem(last=False)

    def memorize_experience(self, vehicle_id, vehicle_state, a):
        t = self.feature_constructor.get_current_time()
        f = self.feature_constructor.get_current_fingerprint()
        l = mesh.convert_lonlat_to_xy(vehicle_state.lon, vehicle_state.lat)
        last_state_action = self.last_state_actions.get(vehicle_id, None)

        if last_state_action is not None:
            current_state = (t, f, l)
            reward = self.rewards[vehicle_id]

            if len(self.experience_memory) > settings.MAX_MEMORY_SIZE:
                self.experience_memory.pop(0)
            self.experience_memory.append((last_state_action, current_state, reward))

        self.rewards[vehicle_id] = 0
        self.last_state_actions[vehicle_id] = (t, f, l, a)


    def train_network(self, batch_size, n_iterations):
        loss_sum = 0
        q_max_sum = 0
        for _ in range(n_iterations):
            s_batch = []
            y_batch = []
            for _ in range(batch_size):
                s, y = self.replay_memory()
                s_batch.append(s)
                y_batch.append(y)

            # print(np.percentile(y_batch, [5, 25, 50, 75, 95]))
            loss_sum += self.q_network.fit(s_batch, y_batch)
            q_max_sum += np.mean(y_batch)
        self.q_network.run_cyclic_updates()
        return loss_sum / n_iterations, q_max_sum / n_iterations


    def replay_memory(self, max_retry=100):
        for _ in range(max_retry):
            num = np.random.randint(0, len(self.experience_memory) - 1)
            state_action, next_state, reward = self.experience_memory[num]
            t, f, l, a = state_action
            next_t, next_f, next_l = next_state
            if not (t in self.supply_demand_history and next_t in self.supply_demand_history):
                self.experience_memory.pop(num)
                continue

            s_M, s_x = self.feature_constructor.construct_state_feature(t, f, l, self.supply_demand_history[t])
            a_M, a_x = self.feature_constructor.construct_action_feature(t, l, self.supply_demand_history[t], a)
            s = (s_M + a_M, s_x + a_x)
            next_s, _ = self.feature_constructor.construct_features(next_t, next_f, next_l, self.supply_demand_history[next_t])
            target_value = self.q_network.compute_target_value(next_s)
            discount_factor = settings.GAMMA ** int((next_t - t) / 60)
            y = reward + discount_factor * target_value
            return s, y

        raise Exception
