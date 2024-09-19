import json
import logging
from typing import Tuple, Optional
import torch
from gym import make
import retro
import numpy as np
import os
from dataclasses import dataclass

from .A3CModel import A3Clstm
from .SharedOptimizers import SharedLrSchedAdam

def setup_logger(logger_name: str, log_file: str, level=logging.INFO):
    """Initialize and configure logger."""
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

def read_config(file_path: str) -> dict:
    """Read JSON configuration from a file."""
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config

def ensure_shared_grads(model: torch.nn.Module, shared_model: torch.nn.Module):
    """Copy gradients from model to shared_model."""
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            continue
        shared_param._grad = param.grad

class AtariRescale:
    def __init__(self, env, env_conf):
        self.env = env
        self.conf = env_conf
        self.observation_space = env.observation_space

    def preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess the game frame."""
        # Example preprocessing; adjust based on your needs
        frame = frame[self.conf["crop1"]:self.conf["crop2"], :]
        frame = retro.to_rgb(frame)
        frame = frame.astype(np.float32) / 255.0
        return torch.from_numpy(frame).permute(2, 0, 1)  # Channels first

    def reset(self) -> torch.Tensor:
        state = self.env.reset()
        return self.preprocess(state)

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, dict]:
        state, reward, done, info = self.env.step(action)
        return self.preprocess(state), reward, done, info

def atari_env(env_id: str, env_conf: dict):
    """Create and configure the Atari environment using gym-retro."""
    env = retro.make(game=env_id)
    env = AtariRescale(env, env_conf)
    return env

@dataclass
class Args:
    environment: str
    config: str
    log_dir: str
    load_model: bool
    load_model_dir: str
    save_model_dir: str
    seed: int
    max_episode_length: int
    num_steps: int
    num_workers: int
    learning_rate: float
    gamma: float
    tau: float
    shared_optimizer: bool
    save_score_level: int
    check_lives: bool
    env_config: str

    @classmethod
    def from_dict(cls, args_dict: dict) -> 'Args':
        return cls(
            environment=args_dict.get('environment', ''),
            config=args_dict.get('config', ''),
            log_dir=args_dict.get('log_dir', ''),
            load_model=args_dict.get('load_model', False),
            load_model_dir=args_dict.get('load_model_dir', ''),
            save_model_dir=args_dict.get('save_model_dir', ''),
            seed=args_dict.get('seed', 0),
            max_episode_length=args_dict.get('max_episode_length', 0),
            num_steps=args_dict.get('num_steps', 0),
            num_workers=args_dict.get('num_workers', 0),
            learning_rate=args_dict.get('learning_rate', 0.0),
            gamma=args_dict.get('gamma', 0.0),
            tau=args_dict.get('tau', 0.0),
            shared_optimizer=args_dict.get('shared_optimizer', False),
            save_score_level=args_dict.get('save_score_level', 0),
            check_lives=args_dict.get('check_lives', False),
            env_config=args_dict.get('env_config', 'settings.json')
        )

def load_arguments(args_dict: dict) -> Tuple[dict, torch.nn.Module, Optional[SharedLrSchedAdam]]:
    """Load and initialize arguments, shared models, and optimizers."""
    undo_logger_setup()
    torch.set_default_dtype(torch.float32)
    torch.set_default_device(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    args = Args.from_dict(args_dict)
    torch.manual_seed(args.seed)

    setup_json = read_config(args.env_config)
    env_conf = setup_json.get(args.config, setup_json.get(args.environment, {}))

    shared_model = A3Clstm(env_conf.get('observation_space', 3), env_conf.get('action_space', 4))
    if args.load_model:
        model_filename = f"{args.environment}.dat"
        model_path = os.path.join(args.load_model_dir, model_filename)
        os.makedirs(args.load_model_dir, exist_ok=True)
        if os.path.exists(model_path):
            shared_model.load_state_dict(torch.load(model_path, weights_only=True))
            logging.info(f"Loaded model from {model_path}")
        else:
            logging.warning(f"Model file {model_path} does not exist. Training from scratch.")
            shared_model.apply(init_weights)
    shared_model.share_memory()

    if args.shared_optimizer:
        optimizer = SharedLrSchedAdam(shared_model.parameters(), lr=args.learning_rate)
        optimizer.share_memory()
    else:
        optimizer = None

    return env_conf, shared_model, optimizer

def undo_logger_setup():
    """Undo any existing logger setup to prevent duplicate logs."""
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()

def init_weights(m: torch.nn.Module):
    """Initialize weights of the model."""
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.LSTMCell):
        for name, param in m.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                torch.nn.init.constant_(param, 0)