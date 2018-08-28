# Navigation 
## A Deep Reinforcement Learning Agent for Navigation using the Unity Machine Learning Agents Toolkit.

In this project, I will demonstrate a Deep Q Learning approach to train an agent to solve the provided Banana navigation environment.

##### Watch the trained agent in action.

The trained agent to navigates (and collects bananas!) in a large, square world.

[![Alt text](https://img.youtube.com/vi/EwCTUsLiWcg/0.jpg)](https://www.youtube.com/watch?v=EwCTUsLiWcg)


##### This project contains:
* Functional, well-documented, and organized code for training the agent implemented in PyTorch and Python 3
* Saved model weights for my successful agent

### A Note on Unity

The **Unity Machine Learning Agents Toolkit (ML-Agents)** is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents. Agents can be trained using reinforcement learning, imitation learning, neuroevolution, or other machine learning methods through a simple-to-use Python API.

## Project Details

In this task, I trained an agent to navigate (and collect bananas!) in a large, square world.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of my agent was to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, my agent had to get an average score of +13 over 100 consecutive episodes.

## Getting Started

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
2. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
	```bash
	git clone https://github.com/bohoro/Navigation.git
	cd Navigation/python
	pip install .
	```
3. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

4. Place the file in the DRLND GitHub repository, in the `Navigation/` folder, and unzip (or decompress) the file. 

## Instructions

1. Run the pre-trained agent.  This step will load the saved weights and run 3 episodes of the task.
	```bash
	cd Navigation/
	python run_agent.py
	```
2. Train and agent from scratch
	```bash
	python UnityAgentDriver.py
	```
## More Information

For more information on the underlying technology see the [Report](Report.md).
