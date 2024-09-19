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
from reinforcement_learninga3c.utils import ensure_shared_grads, atari_env, load_arguments, Args

def train(rank: int, args: Args, shared_model: torch.nn.Module, optimizer: optim.Optimizer, env_conf: dict):
    torch.manual_seed(args.seed + rank)
    env = atari_env(args.environment, env_conf)
    if optimizer is None:
        optimizer = SharedLrSchedAdam(shared_model.parameters(), lr=args.learning_rate)

    env.env.seed(args.seed + rank)
    player = Agent(model=A3Clstm(env.observation_space.shape[0], env.action_space), env=env, args=args)
    player.model.train()

    while True:
        player.model.load_state_dict(shared_model.state_dict())
        for step in range(args.num_steps):
            player.action_train()
            if args.check_lives:
                player.check_state()
            if player.done:
                break

        if player.done:
            player.eps_len = 0
            player.current_life = 0
            state = player.env.reset()
            player.state = state

        R = torch.zeros(1, 1)
        if not player.done:
            with torch.no_grad():
                value, _, _ = player.model(player.state.unsqueeze(0), (player.hx, player.cx))
                R = value

        player.values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(player.rewards))):
            R = args.gamma * R + player.rewards[i]
            advantage = R - player.values[i]
            value_loss += 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation
            delta_t = player.rewards[i] + args.gamma * player.values[i + 1].data - player.values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss -= player.log_probs[i] * gae + 0.01 * player.entropies[i]

        optimizer.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(player.model.parameters(), 40)
        ensure_shared_grads(player.model, shared_model)
        optimizer.step()
        player.clear_actions()
        
        
def train_with_args(args_dict: dict):
    env_conf, shared_model, optimizer = load_arguments(args_dict)
    args = Args.from_dict(args_dict)

    processes = []

    for rank in range(args.num_workers):
        p = Process(target=train, args=(rank, args, shared_model, optimizer, env_conf))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    model_filename = f"{args.environment}.dat"
    model_path = os.path.join(args.save_model_dir, model_filename)
    torch.save(shared_model.state_dict(), model_path)



