# Navigation 
## A Deep Reinforcement Learning Agent for Navigation using the Unity Machine Learning Agents Toolkit.

### Project Report

In this project, I will demonstrate a Deep Q Learning approach to train an agent to solve the provided Banana navigation environment.  This report will elborate on the learning algorithm used, a plot of the scores (rewards) during training, as well as ideas for future work.  

If you would like to install and run the code see the installation steps [here](README.md).

## Learning Algorithm

The reinforcement learning problem as specificfied by Richard S. Sutton and Andrew G. Barto, [Reinforcement Learning:
An Introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)

![RL](https://github.com/bohoro/Navigation/raw/master/media/sutton.png)

In this project the agent takes actions on the banana enviroment and recieves a new state and a reward. 

* A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of my agent was to collect as many yellow bananas as possible while avoiding blue bananas.

* The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available

* The learning algorith used in this project was a double DQN with an optional experimental version of Prioritized Experience Replay.  


It was directly derived from the following works:

* [Human-level control through deep reinforcement
learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
* [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
* [
Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)

Here is the TL;DR summary of the papers and the use of their techniques in the project.

The core algorith is that of a Deep Q-Netowk (DQN).  DQN   

```
Training Agent
Episode 100	Average Score: 0.244
Episode 200	Average Score: 1.92
Episode 300	Average Score: 5.98
Episode 400	Average Score: 9.50
Episode 500	Average Score: 12.33
Episode 596	Average Score: 14.03
Environment solved in 496 episodes!	Average Score: 14.03
```

## Plot of Rewards

Please find below the plot of the training episodes.  The top plot is the 100 episode movign average while the bottom plot is the raw scores by episode.  

![Plot](https://github.com/bohoro/Navigation/raw/master/media/EpisodePlot.jpg)


## Ideas for Future Work

##### to do
