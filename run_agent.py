from unityagents import UnityEnvironment
import numpy as np
import torch
from dqn_agent import Agent

env = UnityEnvironment(file_name="./Banana.app")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=False)[brain_name]


def getEnvInfo(env_info):
    next_state = env_info.vector_observations[0]   # get the next state
    reward = env_info.rewards[0]                   # get the reward
    done = env_info.local_done[0]
    return(next_state, reward, done)


action_size = brain.vector_action_space_size
state_size = len(env_info.vector_observations[0])
agent = Agent(state_size=state_size, action_size=action_size, seed=0)

# load the weights from file
print("Loading Model...", end="")
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
print("Compelete")

state = env_info.vector_observations[0]            # get the current state
score = 0                                          # initialize the score
for i_episode in range(1, 4):
    while True:
        action = agent.act(state)        # select an action
        env_info = env.step(action)[brain_name]        # action to the env
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # episode finished?
        score += reward                                # update the score
        state = next_state                             # state to next ts
        if done:                                       # exit loop if finished
            break
    env_info = env.reset(train_mode=False)[brain_name]
    print("Episode {} Score: {}".format(i_episode, score))

env.close()
