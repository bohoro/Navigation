from collections import namedtuple, deque
import numpy as np
import torch
import random

A_REPLAY = 0.7

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences
                       if e is not None])).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences
                       if e is not None])).long().to(device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences
                       if e is not None])).float().to(device)
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences
                       if e is not None])).float().to(device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(
                np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

# Experimental Prioritized Replay Buffer - not used in final solution
class PrioritizedReplayBuffer:
    """"Fixed-size buffer to store prioritized experience tuples
          with priority."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """"Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.probs = []
        self.probs_dirty_flag = True

    def add(self, state, action, reward, next_state, done, priority=.999):
        """Add a new experience to memory."""
        self.probs_dirty_flag = True
        e = [
            self.experience(state, action, reward, next_state, done), priority
        ]
        self.memory.append(e)

    def update(self, state, action, reward, next_state, done, priority):
        """Update the experience in memory."""
        self.probs_dirty_flag = True
        key_found = False
        key = self.experience(state.numpy(), action.numpy(), reward.numpy(),
                              next_state.numpy(), done.numpy())
        for e in self.memory:
            if np.allclose(e[0][0], key[0], atol=.000001):
                e[1] = priority.detach().item()
                key_found = True
        if key_found is False:
            print("Error updating Key")

    def print_me(self):
        for e in self.memory:
            print(e)

    def sample(self):
        """Prioritized sample a batch of experiences from memory."""
        k = self.batch_size

        if self.probs_dirty_flag is True:
            self.probs = []
            for train in self.memory:
                self.probs.append(train[1])
            self.probs = np.array(self.probs)**A_REPLAY
            self.probs = self.probs / np.sum(self.probs)
            self.probs_dirty_flag = False

        samp = np.random.choice(
            range(0, len(self.memory)), size=k, p=self.probs)
        experiences = [self.memory[i] for i in samp]
        a_probs = [self.probs[i] for i in samp]

        states = torch.tensor(
            [e[0].state for e in experiences if e is not None]).float()
        actions = torch.unsqueeze(
            torch.tensor(
                [int(e[0].action) for e in experiences if e is not None]), 1)
        rewards = torch.unsqueeze(
            torch.tensor([e[0].reward for e in experiences
                          if e is not None]).float(), 1)
        next_states = torch.tensor(
            np.vstack([e[0].next_state for e in experiences
                       if e is not None])).float()
        dones = torch.unsqueeze(
            torch.tensor(
                [e[0].done for e in experiences if e is not None]).float(), 1)

        return (states, actions, rewards, next_states, dones, a_probs)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
