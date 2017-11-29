import torch
from torch.multiprocessing import Process
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable


import torch.optim as optim


class SharedRMSprop(optim.RMSprop):
    """Implements RMSprop algorithm with shared states.
    """

    def __init__(self,
                 params,
                 lr=7e-4,
                 alpha=0.99,
                 eps=0.1,
                 weight_decay=0,
                 momentum=0,
                 centered=False):
        super(SharedRMSprop, self).__init__(params, lr, alpha, eps,
                                            weight_decay, momentum, centered)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['grad_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['square_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['momentum_buffer'] = p.data.new().resize_as_(
                    p.data).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['square_avg'].share_memory_()
                state['step'].share_memory_()
                state['grad_avg'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(1 - alpha, grad)
                    avg = square_avg.addcmul(
                        -1, grad_avg, grad_avg).sqrt().add_(group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])

                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).addcdiv_(grad, avg)
                    p.data.add_(-group['lr'], buf)
                else:
                    p.data.addcdiv_(-group['lr'], grad, avg)

        return loss