# import sys
import os
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
from experiment import Experiment
from agent.matching_policy import GreedyMatchingPolicy, RoughMatchingPolicy
from dqn.dqn_policy import DQNDispatchPolicy, DQNDispatchPolicyLearner
from dqn.settings import BATCH_SIZE, NUM_ITERATIONS, NUM_SUPPLY_DEMAND_HISTORY, FLAGS
from config.settings import TIMESTEP, DEFAULT_LOG_DIR
from common.time_utils import get_local_datetime


def setup_base_log_dir(base_log_dir):
    base_log_path = "./logs/{}".format(base_log_dir)
    if not os.path.exists(base_log_path):
        os.makedirs(base_log_path)
    for dirname in ["sim"]:
        p = os.path.join(base_log_path, dirname)
        if not os.path.exists(p):
            os.makedirs(p)
    if FLAGS.train:
        for dirname in ["networks", "summary", "memory"]:
            p = os.path.join(base_log_path, dirname)
            if not os.path.exists(p):
                os.makedirs(p)

    if os.path.exists(DEFAULT_LOG_DIR):
        os.unlink(DEFAULT_LOG_DIR)
    os.symlink(base_log_dir, DEFAULT_LOG_DIR)

if __name__ == '__main__':
    setup_base_log_dir(FLAGS.tag)

    num_simulation_steps = int(60 * 60 * 24 * FLAGS.days / TIMESTEP)
    num_vehicles = FLAGS.vehicles

    start_time = FLAGS.start_time + int(60 * 60 * 24 * FLAGS.start_offset / TIMESTEP)
    print("Start Datetime: {}".format(get_local_datetime(start_time)))
    end_time = start_time + num_simulation_steps * TIMESTEP
    print("End Datetime  : {}".format(get_local_datetime(end_time)))

    if FLAGS.train:
        print("Set training mode")
        dispatch_policy = DQNDispatchPolicyLearner()
        dispatch_policy.build_q_network(load_network=FLAGS.load_network)

        if FLAGS.load_memory:
            dispatch_policy.load_experience_memory(FLAGS.load_memory)

        if FLAGS.pretrain > 0:
            for i in range(FLAGS.pretrain):
                average_loss, average_q_max = dispatch_policy.train_network(BATCH_SIZE, NUM_ITERATIONS)
                print("iterations : {}, average_loss : {:.3f}, average_q_max : {:.3f}".format(
                    i, average_loss, average_q_max), flush=True)
                dispatch_policy.q_network.write_summary(average_loss, average_q_max)

    else:
        dispatch_policy = DQNDispatchPolicy()
        dispatch_policy.build_q_network(load_network=FLAGS.load_network)


    if FLAGS.use_osrm:
        matching_policy = GreedyMatchingPolicy()
    else:
        matching_policy = RoughMatchingPolicy()

    dqn_exp = Experiment(start_time, TIMESTEP, dispatch_policy, matching_policy)
    dqn_exp.reset()
    if FLAGS.train:
        dispatch_policy.reset()
    dqn_exp.populate_vehicles(n_vehicles=num_vehicles)


    epoch = 3600 * 24 / TIMESTEP
    for i in range(num_simulation_steps):
        dqn_exp.step(verbose=FLAGS.verbose)

        if i == 0:
            continue

        if i % int(3600 / TIMESTEP) == 0:
            print("Elapsed : {:.0f} hours".format(i * TIMESTEP / 3600.0), flush=True)

        if i % epoch == 0:
            dqn_exp.simulator.log_score()

        if FLAGS.train and (i % (epoch * 7) == 0 or i == num_simulation_steps - 1):
            print("Dumping experience memory as pickle...")
            dispatch_policy.dump_experience_memory()