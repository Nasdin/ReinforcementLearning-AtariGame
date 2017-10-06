# RL A3C Pytorch

![A3C LSTM playing Breakout-v0](https://github.com/Nasdin/ReinforcementLearning-AtariGame/blob/master/demo/Breakout.gif) ![A3C LSTM playing SpaceInvadersDeterministic-v3](https://github.com/Nasdin/ReinforcementLearning-AtariGame/blob/master/demo/SpaceInvaders.gif) ![A3C LSTM playing MsPacman-v0](https://github.com/Nasdin/ReinforcementLearning-AtariGame/blob/master/demo/MsPacman.gif) ![A3C LSTM playing BeamRider-v0](https://github.com/Nasdin/ReinforcementLearning-AtariGame/blob/master/demo/BeamRider.gif) ![A3C LSTM playing Seaquest-v0](https://github.com/Nasdin/ReinforcementLearning-AtariGame/blob/master/demo/Seaquest.gif)

# Ipython Notebook Environment 
	Inputs are changed in the Jupyter Notebook

## A3C LSTM, Reinforcement Learning Implementation

Reinforcement learning using Asynchronous Advantage Actor-Critic (A3C) in Pytorch an algorithm from Google Deep Mind's paper "Asynchronous Methods for Deep Reinforcement Learning."

# OpenAI Gym Atari Environment & OpenAI Universe

Implementation of A3C LSTM model trained it.

So far model currently has shown the best performance I have seen for atari game environments.  Included in repo are trained models for SpaceInvaders-v0, MsPacman-v0, Breakout-v0, BeamRider-v0, Pong-v0, Seaquest-v0 and Asteroids-v0 
## Optimizers and Shared optimizers/statitics

#### RMSProp
#### Adam
	Both Shared and non shared available for Adam
In GYM atari the agents randomly repeat the previous action with probability 0.25 and there is time/step limit that limits performance.


## Requirements

- Python 2.7+
- Openai Gym and Universe
- Pytorch

## Training
*When training model it is important to limit number of worker threads to number of cpu cores available as too many threads (e.g. more than one thread per cpu core available) will actually be detrimental in training speed and effectiveness*



![A3C LSTM playing Pong-v0](https://github.com/Nasdin/ReinforcementLearning-AtariGame/demo/Pong.gif)

