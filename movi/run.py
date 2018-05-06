import argparse
from experiment import Experiment
from agent.matching_policy import RoughMatchingPolicy
from dqn.dqn_policy import DQNDispatchPolicy, DQNDispatchPolicyLearner
from dqn.settings import BATCH_SIZE, NUM_ITERATIONS
from config.settings import NUM_SIMULATION_STEPS, NUM_VEHICLES, TIMESTEP, START_TIME, NUM_DRY_RUN
from common.time_utils import get_local_datetime

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", action='store_true', help="use request pattern for demand generation")
    parser.add_argument("--train", action='store_true', help="run training dqn network")
    parser.add_argument("--load", action='store_true', help="load saved dqn network")
    parser.add_argument("--verbose", action='store_true', help="print log verbosely")
    parser.add_argument("--pretrain", type=int, default=0, help="run N pretraining steps using pickled experience memory")
    args = parser.parse_args()


    print("Start Datetime: {}".format(get_local_datetime(START_TIME)))
    end_time = START_TIME + NUM_SIMULATION_STEPS * TIMESTEP
    print("End Datetime  : {}".format(get_local_datetime(end_time)))

    if args.train:
        print("Set training mode")
        dispatch_policy = DQNDispatchPolicyLearner()
        dispatch_policy.build_q_network(load_network=args.load)

    else:
        dispatch_policy = DQNDispatchPolicy()
        if args.load:
            dispatch_policy.build_q_network(load_network=args.load)

    if args.train and args.pretrain > 0:
        dispatch_policy.load_experience_memory()
        for i in range(args.pretrain):
            average_loss, average_q_max = dispatch_policy.train_network(BATCH_SIZE, NUM_ITERATIONS)
            print("iterations : {}, average_loss : {:.3f}, average_q_max : {:.3f}".format(i, average_loss, average_q_max))
            dispatch_policy.q_network.write_summary(average_loss, average_q_max)

    matching_policy = RoughMatchingPolicy()
    dqn_exp = Experiment(START_TIME, TIMESTEP, dispatch_policy, matching_policy, use_pattern=args.pattern)
    dqn_exp.init_vehicles(n_vehicles=NUM_VEHICLES)
    dqn_exp.dry_run(NUM_DRY_RUN)

    for i in range(NUM_SIMULATION_STEPS):
        if i % int(3600 / TIMESTEP) == 0:
            print("Elapsed : {:.0f} hours".format(i * TIMESTEP / 3600.0))
        dqn_exp.step(verbose=args.verbose)

    if args.train:
        print("Dumping experience memory as pickle...")
        dispatch_policy.dump_experience_memory()
