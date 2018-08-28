# Navigation 
## A Deep Reinforcement Learning Agent for Navigation using the Unity Machine Learning Agents Toolkit.

### Project Report

In this project, I will demonstrate a Deep Q Learning approach to train an agent to solve the provided Banana navigation environment.  This report will elaborate on the learning algorithm used, a plot of the scores (rewards) during training, as well as ideas for future work.  

If you would like to install and run the code see the installation steps [here](README.md).

## Learning Algorithm

The reinforcement learning problem as specified by Richard S. Sutton and Andrew G. Barto, [Reinforcement Learning:
An Introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)

![RL](https://github.com/bohoro/Navigation/raw/master/media/sutton.png)

In this project, the agent takes actions on the banana environment and receives a new state and a reward. 

* A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of my agent was to collect as many yellow bananas as possible while avoiding blue bananas.
* The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. 
* Given this information, the agent has to learn how to best select actions. Four discrete actions are available, move forward, move backward, turn left and turn right.

The task is episodic, and to solve the environment, my agent had to get an average score of +13 over 100 consecutive episodes.

#### Double DQN 

The learning algorithm used in this project was a double DQN with an optional experimental version of Prioritized Experience Replay.  

The end learning algorithm derived from the following works:

* [1 - Human-level control through deep reinforcement
learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
* [2 - Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
* [3 -
Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)

Here is the TL;DR summary of the papers and the use of their techniques in the project.

The core algorithm is that of a Deep Q-Network (DQN).  My DQN combines Reinforcement Learning with Deep Learning with fixed targets and experience replay per the DeepMind research[1].  In addition, it adds Double Q-learning component to avoid overestimates in my action values[2].  Finally, an optional experimental version of Prioritized Experience Replay component was implemented by not used in the final solution[3].

At the core of the learning algorithm was a neural network.   It contained 4 Linear layers witrh output of 128, 64, 32, and 4.  All non-final layers used relu activation.  

#### Hyper Parameters used

The agent used the following paramaeters and hyper parameters.  

The size of the buffer for replay experiences
* BUFFER_SIZE = int(1e5)  

The minibatch size used for learning from expereince
* BATCH_SIZE = 64

The discpount factor for the bellman equation.
* GAMMA = 0.99

Interpolation parameter used in the soft update
* TAU = 1e-3              

Learning rate used in the Adam optimizer
* LR = 5e-4

If enough samples are in the buffer, this is how often the learning stpe is executed
* UPDATE_EVERY = 4        

Training output is below.

```
Training Agent
Episode 100    Average Score: 0.244
Episode 200    Average Score: 1.92
Episode 300    Average Score: 5.98
Episode 400    Average Score: 9.50
Episode 500    Average Score: 12.33
Episode 596    Average Score: 14.03
Environment solved in 496 episodes!    Average Score: 14.03
```

## Plot of Rewards

Please find below the plot of the training episodes.  The top plot is the 100 episode moving average while the bottom plot is the raw scores by episode.  

![Plot](https://github.com/bohoro/Navigation/raw/master/media/EpisodePlot.jpg)


## Ideas for Future Work

* Implement a more performannt version of Prioritized Experience Replay.  
* Add Dueling networks, see [Dueling Network Architectures for Deep Reinforcement Learning](http://proceedings.mlr.press/v48/wangf16.pdf)
* Add Noisy Networks, see [
Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295)
