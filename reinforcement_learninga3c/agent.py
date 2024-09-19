import torch
from torch.autograd import Variable
import torch.nn.functional as F
import logging
import time

from .A3CModel import A3Clstm
from .SharedOptimizers import SharedLrSchedAdam
from .utils import Args

class Agent:
    def __init__(self, model: A3Clstm, env, args: Args):
        self.model = model
        self.env = env
        self.current_life = 0
        self.state = self.env.reset()
        self.hx = torch.zeros(1, 512, dtype=torch.float32)
        self.cx = torch.zeros(1, 512, dtype=torch.float32)
        self.eps_len = 0
        self.args = args
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.done = True
        self.info = None
        self.reward = 0

    def action_train(self):
        if self.done:
            self.cx = torch.zeros(1, 512, dtype=torch.float32)
            self.hx = torch.zeros(1, 512, dtype=torch.float32)
        
        with torch.no_grad():
            value, logit, (self.hx, self.cx) = self.model(self.state.unsqueeze(0), (self.hx, self.cx))
            prob = F.softmax(logit, dim=1)
            log_prob = F.log_softmax(logit, dim=1)
            entropy = -(log_prob * prob).sum(1)
            action = prob.multinomial(num_samples=1).squeeze(1)
            log_prob = log_prob.gather(1, action.unsqueeze(1)).squeeze(1)
        
        state, self.reward, self.done, self.info = self.env.step(action.item())
        self.state = state
        self.eps_len += 1
        self.done = self.done or self.eps_len >= self.args.max_episode_length
        self.reward = max(min(self.reward, 1), -1)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.rewards.append(self.reward)
        self.entropies.append(entropy)
        return self

    def action_test(self):
        if self.done:
            self.cx = torch.zeros(1, 512, dtype=torch.float32)
            self.hx = torch.zeros(1, 512, dtype=torch.float32)
        
        with torch.no_grad():
            value, logit, (self.hx, self.cx) = self.model(self.state.unsqueeze(0), (self.hx, self.cx))
            prob = F.softmax(logit, dim=1)
            action = prob.argmax(1).item()
        
        state, self.reward, self.done, self.info = self.env.step(action)
        self.state = state
        self.eps_len += 1
        self.done = self.done or self.eps_len >= self.args.max_episode_length
        return self

    def check_state(self):
        if self.current_life > self.info.get('lives', 0):
            self.done = True
        self.current_life = self.info.get('lives', self.current_life)
        return self

    def clear_actions(self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        return self