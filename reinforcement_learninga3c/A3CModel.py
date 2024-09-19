import torch
import torch.nn as nn

class A3Clstm(nn.Module):
    def __init__(self, input_channels: int, action_space: int):
        super(A3Clstm, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU()
        )
        self.lstm = nn.LSTMCell(32 * 9 * 9, 512)
        self.actor = nn.Linear(512, action_space)  # Changed from action_space.n to action_space
        self.critic = nn.Linear(512, 1)

    def forward(self, x, hidden):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        hx, cx = hidden
        hx, cx = self.lstm(x, (hx, cx))
        value = self.critic(hx)
        logit = self.actor(hx)
        return value, logit, (hx, cx)


