import json
import logging
import os
import time
import torch
from torch.autograd import Variable
from torch.multiprocessing import Process
import torch.nn.functional as F
import torch.optim as optim

from reinforcement_learninga3c.agent import Agent
from reinforcement_learninga3c.A3CModel import A3Clstm
from reinforcement_learninga3c.SharedOptimizers import SharedLrSchedAdam
from reinforcement_learninga3c.utils import setup_logger, read_config, ensure_shared_grads, atari_env, load_arguments, Args

def test(args: Args, shared_model: torch.nn.Module, env_conf: dict, render: bool = False):
    logger = setup_logger(f"{args.environment}_log", f"{args.log_dir}{args.environment}_log")
    for k, v in args.__dict__.items():
        logger.info(f"{k}: {v}")

    torch.manual_seed(args.seed)
    env = atari_env(args.environment, env_conf)
    reward_sum = 0
    start_time = time.time()
    num_tests = 0
    reward_total_sum = 0
    player = Agent(model=A3Clstm(env.observation_space.shape[0], env.action_space), env=env, args=args)
    player.model.load_state_dict(shared_model.state_dict())
    player.model.eval()

    os.makedirs(args.save_model_dir, exist_ok=True)

    while True:
        if player.done:
            player.model.load_state_dict(shared_model.state_dict())
        if render:
            env.env.render()
        player.action_test()
        reward_sum += player.reward

        if player.done:
            num_tests += 1
            player.current_life = 0
            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests
            logger.info(
                f"Time {time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - start_time))}, "
                f"episode reward {reward_sum}, episode length {player.eps_len}, reward mean {reward_mean:.4f}"
            )

            if reward_sum > args.save_score_level:
                player.model.load_state_dict(shared_model.state_dict())
                state_to_save = player.model.state_dict()
                torch.save(state_to_save, f"{args.save_model_dir}{args.environment}.dat")

            reward_sum = 0
            player.eps_len = 0
            state = player.env.reset()
            player.state = state
            time.sleep(1)  # Reduced sleep time for practicality
            
def test_with_args(args_dict: dict):
    env_conf, shared_model, optimizer = load_arguments(args_dict)
    args = Args.from_dict(args_dict)

    p = Process(target=test, args=(args, shared_model, env_conf))
    p.start()

    p.join()


