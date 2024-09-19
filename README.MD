# Reinforcement Learning A3C with Gym-Retro
## Overview

This project implements the Asynchronous Advantage Actor-Critic (A3C) algorithm with LSTM for scalable reinforcement learning using gym-retro environments. It leverages PyTorch 2 for model development and training.

### Features

- **A3C with LSTM:** Enhances the agent's ability to handle partial observability.
- **Gym-Retro Integration:** Supports a wide range of retro games for training.
- **PyTorch 2 Compatibility:** Utilizes the latest PyTorch features for optimal performance.
- **Multiprocessing Support:** Efficiently trains multiple agents in parallel.
- **Modular Architecture:** Clean separation of utilities, models, and agent logic.
- **Structured Examples:** Dedicated training and testing scripts for each game.



# Required python version = Python 3.8

![A3C LSTM playing SpaceInvadersDeterministic-v3](https://github.com/Nasdin/ReinforcementLearning-AtariGame/blob/master/img/SpaceInvaders.gif) ![A3C LSTM playing Breakout-v0](https://github.com/Nasdin/ReinforcementLearning-AtariGame/blob/master/img/Breakout.gif) 
## Asynchronous Advantage Actor-Critic (A3C) Reinforcement Learning Implementation
### Long Short Term Recurrent Neural Network with Pytorch

An algorithm from Google Deep Mind's paper "Asynchronous Methods for Deep Reinforcement Learning."<br>
https://arxiv.org/pdf/1602.01783.pdf

### Using Google DeepMind's Algorithm. 

Asynchronous Advantage Actor-Critic (A3C)

![A3C LSTM playing Seaquest-v0](https://github.com/Nasdin/ReinforcementLearning-AtariGame/blob/master/img/Seaquest.gif) ![A3C LSTM playing BeamRider-v0](https://github.com/Nasdin/ReinforcementLearning-AtariGame/blob/master/img/BeamRider.gif) 
##### Description
The A3C algorithm was released by Google’s DeepMind group earlier this year, and it made a splash by essentially obsoleting DQN. It was faster, simpler, more robust, and able to achieve much better scores on the standard battery of Deep RL tasks. On top of all that it could work in continuous as well as discrete action spaces. Given this, it has become the go-to Deep RL algorithm for new challenging problems with complex state and action spaces


    
<a href= "https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2" >Medium Article explaining A3c reinforcement learning </a>
<br>
![A3C LSTM playing MsPacman-v0](https://github.com/Nasdin/ReinforcementLearning-AtariGame/blob/master/img/MsPacman.gif)
## The Actor-Critic Structure
<img src = "img/A3CStructure.png">

## Many workers training and learning concurrently, and then updates global network with gradients
### Process Flow
<img src = "img/A3CProcessFlow.png">

    
### Long Short Term Memory Recurrent Neural Nets
	Implemented using Pytorch
<a href= "http://colah.github.io/posts/2015-08-Understanding-LSTMs/" >Understanding LSTM Post </a>


## Trained models

  Trained models are generated when you run through a full training episode for the sim. Continous running will update the model with new training. The L(Load) parameter is set to false in the demo, When you have trained data where it can pick up from, then set it to true.

*In gym atari the agents randomly repeat the previous action with probability 0.25 and there is time/step limit that limits performance. You can adjust these parameters.*

## Optimizers and Shared optimizers/statistics

#### RMSProp

RMSprop is an unpublished, adaptive learning rate method proposed by Geoff Hinton in Lecture 6e of his Coursera Class.

RMSprop and Adadelta have both been developed independently around the same time stemming from the need to resolve Adagrad's radically diminishing learning rates. RMSprop in fact is identical to the first update vector of Adadelta 

RMSprop as well divides the learning rate by an exponentially decaying average of squared gradients. Hinton suggests γ
to be set to 0.9, while a good default value for the learning rate η is 0.001.

#### Adaptive Moment Estimation (Adam) , Both Shared and non shared available for Adam
is another method that computes adaptive learning rates for each parameter. In addition to storing an exponentially decaying average of past squared gradients vt like Adadelta and RMSprop, Adam also keeps an exponentially decaying average of past gradients mt, similar to momentum:

Adam (short for Adaptive Moment Estimation) is an update to the RMSProp optimizer. In this optimization algorithm, running averages of both the gradients and the second moments of the gradients are used.
	
# WARNING, ROM Files
You must download the ROM files for the games you want to train and test. 
https://retro.readthedocs.io/en/latest/index.html

## Training
It is important to limit number of worker threads to number of cpu cores available in your com.
If you use more than one worker thread than the amount of cpu cores, it will result in poor performance and inefficiency.


## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/reinforcement_learninga3c.git
   cd reinforcement_learninga3c
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Gym-Retro Games:**
   Follow the [gym-retro documentation](https://github.com/openai/retro#installation) to install the desired games.

## Usage

### Training Agents

To train an agent on a specific game (e.g., Mario), navigate to the corresponding example subfolder and run the `train.py` script.
## Examples

bash
python -m examples.128_sine_dot.train
python -m examples.automaton.train

### Testing Agents

To test a trained agent with rendering enabled, navigate to the corresponding example subfolder and run the `test.py` script.

bash
python -m examples.128_sine_dot.test
python -m examples.automaton.test

## Examples

Inside the `examples` directory, each game has its own subfolder containing `train.py` and `test.py` scripts. These scripts are configured to train and test agents on their respective games.

examples/
├── mario/
│ ├── train.py
│ └── test.py
├── sonic/
│ ├── train.py
│ └── test.py


## Project Structure

- `reinforcement_learninga3c/`
  - `agent.py`: Defines the Agent class.
  - `A3CModel.py`: Contains the A3Clstm model architecture.
  - `SharedOptimizers.py`: Implements shared optimizers for multiprocessing.
  - `utils.py`: Utility functions for logging, configuration, and environment setup.
  - `__init__.py`: Initializes the reinforcement_learninga3c package.
  - `examples/`
    - `mario/`
	  - `train.py`: Training script for Mario.
	  - `test.py`: Testing script for Mario.
    - `sonic/`
	  - `train.py`: Training script for Sonic.
	  - `test.py`: Testing script for Sonic.
- `modeldata/`: Directory to store saved model data.
- `settings.json`: Configuration file for various environments.
- `requirements.txt`: Lists project dependencies.
- `README.MD`: Project documentation.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.


# Reinforcement Learning implementation of LSTM with Asynchronous Advantage Actor Critic Algorithm
## Using Pytorch on OpenAI Atari Games
	Using OpenAi Gym and Universe. 
	LSTM(Long Short Term Memory) with Pytorch
	Implementation of Google Deepmind's Asynchronous Advantage Actor-Critic (A3C)
