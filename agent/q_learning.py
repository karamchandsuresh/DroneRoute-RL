import random

import numpy as np


class QLearningAgent:
    """
    Battery-aware Q-Learning agent for DroneRoute RL.

    The Q-table stores values for:
        row × column × battery × action
    """

    def __init__(
        self,
        grid_size,
        max_battery=20,
        num_actions=4,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.01
    ):
        self.grid_size = grid_size
        self.max_battery = max_battery
        self.num_actions = num_actions

        # Q-Learning hyperparameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        # Exploration parameters
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Q-table:
        # row × column × battery level × action
        self.q_table = np.zeros(
            (
                grid_size,
                grid_size,
                max_battery + 1,
                num_actions
            )
        )

    def choose_action(self, state):
        """
        Select an action using epsilon-greedy exploration.
        """

        # Exploration
        if random.random() < self.epsilon:
            return random.randint(
                0,
                self.num_actions - 1
            )

        # Exploitation
        row, col, battery = state

        return int(
            np.argmax(
                self.q_table[
                    row,
                    col,
                    battery
                ]
            )
        )

    def update_q_value(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):
        """
        Update Q-value using the Q-Learning equation.
        """

        row, col, battery = state

        next_row, next_col, next_battery = (
            next_state
        )

        current_q = self.q_table[
            row,
            col,
            battery,
            action
        ]

        if done:
            max_future_q = 0

        else:
            max_future_q = np.max(
                self.q_table[
                    next_row,
                    next_col,
                    next_battery
                ]
            )

        target_q = (
            reward
            + self.discount_factor
            * max_future_q
        )

        new_q = (
            current_q
            + self.learning_rate
            * (
                target_q
                - current_q
            )
        )

        self.q_table[
            row,
            col,
            battery,
            action
        ] = new_q

    def decay_epsilon(self):
        """
        Gradually reduce exploration.
        """

        self.epsilon = max(
            self.min_epsilon,
            self.epsilon
            * self.epsilon_decay
        )