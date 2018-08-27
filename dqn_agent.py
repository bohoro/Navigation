import numpy as np
import random
from model import QNetwork
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network
DOUBLE_DQN_ENABLED = True
PRIORITIZED_REPLY_ENABLED = False
E_REPLAY = 0.01
DEBUG = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size,
                                       seed).to(device)
        # Target or w-
        self.qnetwork_target = QNetwork(state_size, action_size,
                                        seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        if (PRIORITIZED_REPLY_ENABLED):
            self.memory = PrioritizedReplayBuffer(action_size,
                                                  BUFFER_SIZE, BATCH_SIZE,
                                                  seed)
        else:
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE,
                                       BATCH_SIZE, seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.B = .001

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random
            #  subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done)
                tuples
            gamma (float): discount factor
        """
        if (PRIORITIZED_REPLY_ENABLED):
            states, actions, rewards, next_states, dones, a_probs = experiences
        else:
            states, actions, rewards, next_states, dones = experiences

        if (DOUBLE_DQN_ENABLED):
            # We will use the local paramters to get the next best action
            # We will then get the Q(s', a ) from the target
            # Get max predicted Q values (for next states) from target model
            # execute prediction for next states
            with torch.no_grad():
                Q_targets_next = self.qnetwork_local(next_states).detach()
            # Returns the maximum value of each row of the input tensor in the
            # given dimension dim. The second return value is the index
            # location of each maximum value found (argmax).
            Q_targets_next = np.argmax(Q_targets_next, axis=1)
            Q_targets_next_prime = self.qnetwork_target(next_states).detach()
            Q_targets_next = Q_targets_next_prime[list(range(0, len(
                states))), Q_targets_next].reshape(len(states), 1)
        else:
            # Get max predicted Q values (for next states) from target model
            # execute prediction for next states
            Q_targets_next = self.qnetwork_target(next_states)
            # Detaches the Tensor from the graph that created it, making it
            # a leaf.
            Q_targets_next = Q_targets_next.detach()
            # Returns the maximum value of each row of the input tensor in
            # the given dimension dim. The second return value is the index
            # location of each maximum value found (argmax).
            Q_targets_next = Q_targets_next.max(1)[0]
            # Returns a new tensor with a dimension of size one inserted at the
            # specified position.
            # Before squeze  torch.Size([64])
            Q_targets_next = Q_targets_next.unsqueeze(1)
            # After squeze torch.Size([64, 1])

        # Compute Q targets for current states
        # start with the rewards
        Q_targets = rewards
        # gamma * make on next state but only if not done
        Q_targets += (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        # do a forward pass
        Q_expected = self.qnetwork_local(states)
        # Gathers values along an axis specified by dim.
        # Before gather  torch.Size([64, 4])
        Q_expected = Q_expected.gather(1, actions)
        # After gather  torch.Size([64, 1])

        # Calulate the loss
        if (PRIORITIZED_REPLY_ENABLED):
            td_error = (Q_expected - Q_targets).abs_() + E_REPLAY
            impSampleWeigt = torch.tensor(
                ((1 / np.array(a_probs)) * (1 / BUFFER_SIZE))**self.B).float()
            for i in range(len(experiences)):
                self.memory.update(states[i], actions[i], rewards[i],
                                   next_states[i], dones[i], td_error[i])
            loss = F.mse_loss(Q_expected, Q_targets, reduce=False)
            impSampleWeigt = torch.unsqueeze(impSampleWeigt, 1)
            if self.B < 0.998:
                self.B += .001
            loss = torch.mean(loss * torch.tensor(impSampleWeigt).float())
        else:
            # Compute loss
            loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(),
                                             local_model.parameters()):
            target_param.data.copy_(tau * local_param.data +
                                    (1.0 - tau) * target_param.data)
