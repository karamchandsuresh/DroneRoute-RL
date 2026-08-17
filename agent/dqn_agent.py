import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQNetwork(nn.Module):
    """
    Neural network used to approximate Q-values.

    Input:
        [row, column, battery]

    Output:
        Q-value for each action:
        [UP, DOWN, LEFT, RIGHT]
    """

    def __init__(
        self,
        state_size=3,
        action_size=4
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, action_size)
        )

    def forward(self, state):
        """
        Perform a forward pass through the network.
        """
        return self.network(state)


class DQNAgent:
    """
    Deep Q-Network agent for DroneRoute RL.

    Uses:
    - Neural-network Q-value approximation
    - Experience replay
    - Target network
    - Epsilon-greedy exploration
    """

    def __init__(
        self,
        grid_size=5,
        max_battery=20,
        action_size=4,
        learning_rate=0.001,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.01,
        replay_capacity=10000,
        batch_size=64
    ):
        self.grid_size = grid_size
        self.max_battery = max_battery

        self.state_size = 3
        self.action_size = action_size

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.batch_size = batch_size

        # CPU/GPU selection
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # Main network
        self.policy_network = DQNetwork(
            state_size=self.state_size,
            action_size=self.action_size
        ).to(self.device)

        # Target network
        self.target_network = DQNetwork(
            state_size=self.state_size,
            action_size=self.action_size
        ).to(self.device)

        # Initially both networks have identical weights
        self.target_network.load_state_dict(
            self.policy_network.state_dict()
        )

        self.target_network.eval()

        # Adam optimizer
        self.optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=self.learning_rate
        )

        # Mean Squared Error loss
        self.loss_function = nn.MSELoss()

        # Experience replay memory
        self.memory = deque(
            maxlen=replay_capacity
        )

    def normalize_state(self, state):
        """
        Normalize the state values before feeding
        them into the neural network.

        Original:
            (row, column, battery)

        Normalized approximately to:
            0.0 - 1.0
        """

        row, col, battery = state

        normalized_state = np.array(
            [
                row / (self.grid_size - 1),
                col / (self.grid_size - 1),
                battery / self.max_battery
            ],
            dtype=np.float32
        )

        return normalized_state

    def choose_action(self, state):
        """
        Select an action using epsilon-greedy strategy.
        """

        # Exploration
        if random.random() < self.epsilon:
            return random.randrange(
                self.action_size
            )

        # Exploitation
        normalized_state = self.normalize_state(
            state
        )

        state_tensor = torch.tensor(
            normalized_state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():

            q_values = self.policy_network(
                state_tensor
            )

        return int(
            torch.argmax(
                q_values,
                dim=1
            ).item()
        )

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):
        """
        Store one experience in replay memory.
        """

        self.memory.append(
            (
                state,
                action,
                reward,
                next_state,
                done
            )
        )

    def replay(self):
        """
        Train the policy network using a random
        batch sampled from replay memory.
        """

        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(
            self.memory,
            self.batch_size
        )

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for (
            state,
            action,
            reward,
            next_state,
            done
        ) in batch:

            states.append(
                self.normalize_state(state)
            )

            actions.append(action)
            rewards.append(reward)

            next_states.append(
                self.normalize_state(next_state)
            )

            dones.append(done)

        states = torch.tensor(
            np.array(states),
            dtype=torch.float32,
            device=self.device
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long,
            device=self.device
        ).unsqueeze(1)

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=self.device
        )

        next_states = torch.tensor(
            np.array(next_states),
            dtype=torch.float32,
            device=self.device
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=self.device
        )

        # Q-values predicted by policy network
        current_q_values = (
            self.policy_network(states)
            .gather(1, actions)
            .squeeze(1)
        )

        # Q-values for next states from target network
        with torch.no_grad():

            next_q_values = (
                self.target_network(next_states)
                .max(dim=1)[0]
            )

            target_q_values = (
                rewards
                + self.discount_factor
                * next_q_values
                * (1 - dones)
            )

        # Calculate training loss
        loss = self.loss_function(
            current_q_values,
            target_q_values
        )

        # Neural-network update
        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """
        Copy policy-network weights to
        the target network.
        """

        self.target_network.load_state_dict(
            self.policy_network.state_dict()
        )

    def decay_epsilon(self):
        """
        Gradually reduce exploration.
        """

        self.epsilon = max(
            self.min_epsilon,
            self.epsilon
            * self.epsilon_decay
        )