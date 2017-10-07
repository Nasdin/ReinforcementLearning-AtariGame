# Reinforcement Learning implementation using Pytorch on Atari Games

# Ipython Notebook Environment 
	Inputs are changed in the Jupyter Notebook

![A3C LSTM playing Breakout-v0](https://github.com/Nasdin/ReinforcementLearning-AtariGame/blob/master/demo/Breakout.gif) ![A3C LSTM playing MsPacman-v0](https://github.com/Nasdin/ReinforcementLearning-AtariGame/blob/master/demo/MsPacman.gif)

## Environment provided by OpenAI Gym and Universe
![A3C LSTM playing SpaceInvadersDeterministic-v3](https://github.com/Nasdin/ReinforcementLearning-AtariGame/blob/master/demo/SpaceInvaders.gif)![A3C LSTM playing Seaquest-v0](https://github.com/Nasdin/ReinforcementLearning-AtariGame/blob/master/demo/Seaquest.gif)

## Asynchronous Advantage Actor-Critic (A3C) LSTM, Reinforcement Learning Implementation

An algorithm from Google Deep Mind's paper "Asynchronous Methods for Deep Reinforcement Learning."
 ![A3C LSTM playing BeamRider-v0](https://github.com/Nasdin/ReinforcementLearning-AtariGame/blob/master/demo/BeamRider.gif) 

## Trained models

  Trained models are generated when you run through a full training episode for the sim. Continous running will update the model with new training. The L(Load) parameter is set to false in the demo, When you have trained data where it can pick up from, then set it to true.

## Optimizers and Shared optimizers/statistics

#### RMSProp
#### Adam
	Both Shared and non shared available for Adam
In GYM atari the agents randomly repeat the previous action with probability 0.25 and there is time/step limit that limits performance.



## Training
*It is important to limit number of worker threads to number of cpu cores available
More than one thread per cpu core available is detrimental in training speed and effectiveness*



