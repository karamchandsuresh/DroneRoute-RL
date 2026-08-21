import random

import numpy as np


class QLearningAgent:
    """
    Dynamic-obstacle-aware and battery-aware
    Q-Learning agent for DroneRoute RL.

    State:
        (
            row,
            column,
            battery,
            blocked_up,
            blocked_down,
            blocked_left,
            blocked_right
        )

    Q-table dimensions:
        row
        x column
        x battery
        x blocked_up
        x blocked_down
        x blocked_left
        x blocked_right
        x action
    """

    def __init__(
        self,
        grid_size,
        max_battery=20,
        num_actions=4,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.997,
        min_epsilon=0.02,
    ):
        self.grid_size = grid_size
        self.max_battery = max_battery
        self.num_actions = num_actions

        # --------------------------------------------------
        # Q-Learning hyperparameters
        # --------------------------------------------------

        self.learning_rate = (
            learning_rate
        )

        self.discount_factor = (
            discount_factor
        )

        # --------------------------------------------------
        # Exploration parameters
        # --------------------------------------------------

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # --------------------------------------------------
        # Q-table
        # --------------------------------------------------

        self.q_table = np.zeros(
            (
                grid_size,
                grid_size,
                max_battery + 1,
                2,
                2,
                2,
                2,
                num_actions,
            ),
            dtype=np.float32,
        )

    # ==================================================
    # STATE INDEX
    # ==================================================

    def _state_index(
        self,
        state,
    ):
        """
        Convert obstacle-aware state into
        a Q-table index.
        """

        (
            row,
            col,
            battery,
            blocked_up,
            blocked_down,
            blocked_left,
            blocked_right,
        ) = state

        battery = max(
            0,
            min(
                battery,
                self.max_battery,
            ),
        )

        return (
            row,
            col,
            battery,
            blocked_up,
            blocked_down,
            blocked_left,
            blocked_right,
        )

    # ==================================================
    # ACTION SELECTION
    # ==================================================

    def choose_action(
        self,
        state,
    ):
        """
        Select an action using epsilon-greedy
        exploration.
        """

        # Exploration
        if (
            random.random()
            < self.epsilon
        ):
            return random.randrange(
                self.num_actions
            )

        # Exploitation
        state_index = (
            self._state_index(
                state
            )
        )

        return int(
            np.argmax(
                self.q_table[
                    state_index
                ]
            )
        )

    # ==================================================
    # Q-VALUE UPDATE
    # ==================================================

    def update_q_value(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):
        """
        Update Q-value using the
        Q-Learning update equation.
        """

        state_index = (
            self._state_index(
                state
            )
        )

        next_state_index = (
            self._state_index(
                next_state
            )
        )

        current_q = (
            self.q_table[
                state_index
                + (action,)
            ]
        )

        # No future reward is considered
        # after a terminal state.
        if done:
            max_future_q = 0.0

        else:
            max_future_q = float(
                np.max(
                    self.q_table[
                        next_state_index
                    ]
                )
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
            state_index
            + (action,)
        ] = new_q

    # ==================================================
    # EPSILON DECAY
    # ==================================================

    def decay_epsilon(self):
        """
        Gradually reduce exploration.

        A slower decay is used because the
        obstacle configuration changes between
        episodes. This gives the agent more time
        to explore different dynamic states.
        """

        self.epsilon = max(
            self.min_epsilon,
            self.epsilon
            * self.epsilon_decay,
        )