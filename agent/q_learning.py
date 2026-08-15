import random

import numpy as np


class QLearningAgent:
    """
    Q-Learning agent for DroneRoute RL.

    The agent learns the best action for each state by interacting
    with the drone environment and updating a Q-table.
    """

    def __init__(
        self,
        grid_size,
        num_actions=4,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.01
    ):
        # Environment information
        self.grid_size = grid_size
        self.num_actions = num_actions

        # Q-Learning hyperparameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        # Exploration parameters
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Q-table dimensions:
        # grid row × grid column × number of actions
        self.q_table = np.zeros(
            (grid_size, grid_size, num_actions)
        )

    def choose_action(self, state):
        """
        Choose an action using the epsilon-greedy strategy.

        Exploration:
            Choose a random action.

        Exploitation:
            Choose the action with the highest Q-value.
        """

        # Exploration
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)

        # Exploitation
        row, col = state

        return int(np.argmax(self.q_table[row, col]))

    def update_q_value(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):
        """
        Update the Q-value using the Q-Learning update rule.
        """

        row, col = state
        next_row, next_col = next_state

        # Current Q-value
        current_q = self.q_table[row, col, action]

        # If destination is reached, there is no future reward.
        if done:
            max_future_q = 0
        else:
            max_future_q = np.max(
                self.q_table[next_row, next_col]
            )

        # Target value
        target_q = (
            reward
            + self.discount_factor * max_future_q
        )

        # Q-Learning update
        new_q = current_q + self.learning_rate * (
            target_q - current_q
        )

        self.q_table[row, col, action] = new_q

    def decay_epsilon(self):
        """
        Reduce exploration gradually as training progresses.
        """

        self.epsilon = max(
            self.min_epsilon,
            self.epsilon * self.epsilon_decay
        )