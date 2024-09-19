import math
from typing import Callable, List, Optional, Tuple

import torch
from torch.optim import Optimizer, RMSprop


class SharedLrSchedAdam(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        sample_lr: List[float] = [
            0.0001, 0.00009, 0.00008, 0.00007, 0.00006, 0.00005, 0.00004, 0.00003,
            0.00002, 0.00001, 0.000009, 0.000008, 0.000007, 0.000006, 0.000005,
            0.000004, 0.000003, 0.000002, 0.000001
        ],
        lr_update_interval: int = 40_000_000
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)

        self.sample_lr = sample_lr
        self.lr_update_interval = lr_update_interval

        # Initialize step and state for each parameter group
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1, dtype=torch.int64)
                state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if amsgrad:
                    state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
                if group['amsgrad']:
                    state['max_exp_avg_sq'].share_memory_()

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                grads.append(p.grad)
                state = self.state[p]

                # Update learning rate
                step = state['step']
                lr_index = int(step.item() // self.lr_update_interval)
                group['lr'] = self.sample_lr[min(lr_index, len(self.sample_lr) - 1)]

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                if group['amsgrad']:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                state_steps.append(step)

            self._adam_update(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group['amsgrad'],
                beta1=group['betas'][0],
                beta2=group['betas'][1],
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                eps=group['eps'],
            )

        return loss

    def _adam_update(
        self,
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        exp_avgs: List[torch.Tensor],
        exp_avg_sqs: List[torch.Tensor],
        max_exp_avg_sqs: List[torch.Tensor],
        state_steps: List[torch.Tensor],
        amsgrad: bool,
        beta1: float,
        beta2: float,
        lr: float,
        weight_decay: float,
        eps: float,
    ):
        for i, param in enumerate(params):
            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step = state_steps[i]

            bias_correction1 = 1 - beta1 ** step.item()
            bias_correction2 = 1 - beta2 ** step.item()

            if weight_decay != 0:
                grad = grad.add(param, alpha=weight_decay)

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            if amsgrad:
                max_exp_avg_sq = max_exp_avg_sqs[i]
                torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

            step_size = lr / bias_correction1

            param.addcdiv_(exp_avg, denom, value=-step_size)
            step += 1

class SharedRMSProp(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0,
        momentum: float = 0,
        centered: bool = False,
    ):
        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum, centered=centered)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1, dtype=torch.int64)
                state['square_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if group['momentum'] > 0:
                    state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if group['centered']:
                    state['grad_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['square_avg'].share_memory_()
                if group['momentum'] > 0:
                    state['momentum_buffer'].share_memory_()
                if group['centered']:
                    state['grad_avg'].share_memory_()

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            alpha = group['alpha']
            eps = group['eps']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            centered = group['centered']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.to(p.device)  # Ensure gradient is on the same device as parameter

                state = self.state[p]

                state['step'] += 1

                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                state['square_avg'] = state['square_avg'].mul(alpha).addcmul(grad, grad, value=1 - alpha)

                if centered:
                    state['grad_avg'] = state['grad_avg'].mul(alpha).add(grad, alpha=1 - alpha)
                    avg = state['grad_avg'] / (1 - alpha ** state['step'].item())
                else:
                    avg = None

                if momentum > 0:
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(grad)
                    if centered:
                        denom = state['square_avg'].sqrt().add_(eps)
                        denom = denom.add(avg.sqrt())
                        p.addcdiv_(buf, denom, value=-lr)
                    else:
                        denom = state['square_avg'].sqrt().add_(eps)
                        p.addcdiv_(buf, denom, value=-lr)
                else:
                    if centered:
                        denom = state['square_avg'].sqrt().add_(eps)
                        denom = denom.add(avg.sqrt())
                        p.addcdiv_(grad, denom, value=-lr)
                    else:
                        denom = state['square_avg'].sqrt().add_(eps)
                        p.addcdiv_(grad, denom, value=-lr)

        return loss

