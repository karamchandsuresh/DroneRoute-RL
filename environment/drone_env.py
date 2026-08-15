class DroneEnvironment:
    """
    Custom grid-based environment for DroneRoute RL.

    The drone must travel from its starting position to the delivery
    destination while avoiding obstacles.
    """

    def __init__(self, grid_size=5):
        self.grid_size = grid_size

        # Start and destination
        self.start_position = (0, 0)
        self.destination = (grid_size - 1, grid_size - 1)

        # Fixed obstacles
        self.obstacles = [
            (1, 1),
            (2, 2),
            (3, 1)
        ]

        # Current drone position
        self.drone_position = self.start_position

        # Action space
        # 0 = UP
        # 1 = DOWN
        # 2 = LEFT
        # 3 = RIGHT
        self.actions = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1)
        }

    def reset(self):
        """
        Reset the environment for a new episode.
        """
        self.drone_position = self.start_position

        return self.drone_position

    def step(self, action):
        """
        Execute one action in the environment.

        Returns:
            next_state: New drone position
            reward: Reward received after the action
            done: Whether the episode has finished
        """

        if action not in self.actions:
            raise ValueError("Invalid action. Choose 0, 1, 2, or 3.")

        # Get row and column movement for the selected action
        row_change, col_change = self.actions[action]

        current_row, current_col = self.drone_position

        # Calculate proposed new position
        new_row = current_row + row_change
        new_col = current_col + col_change

        new_position = (new_row, new_col)

        # --------------------------------------------------
        # Case 1: Drone tries to leave the grid
        # --------------------------------------------------
        if not self._is_inside_grid(new_position):
            reward = -10
            done = False

            # Drone stays in the same position
            return self.drone_position, reward, done

        # --------------------------------------------------
        # Case 2: Drone hits an obstacle
        # --------------------------------------------------
        if new_position in self.obstacles:
            reward = -20
            done = False

            # Drone stays in the same position
            return self.drone_position, reward, done

        # Move drone to valid position
        self.drone_position = new_position

        # --------------------------------------------------
        # Case 3: Drone reaches destination
        # --------------------------------------------------
        if self.drone_position == self.destination:
            reward = 100
            done = True

            return self.drone_position, reward, done

        # --------------------------------------------------
        # Case 4: Normal movement
        # --------------------------------------------------
        reward = -1
        done = False

        return self.drone_position, reward, done

    def _is_inside_grid(self, position):
        """
        Check whether a position exists inside the grid.
        """

        row, col = position

        return (
            0 <= row < self.grid_size
            and 0 <= col < self.grid_size
        )

    def render(self):
        """
        Display the current environment in the terminal.
        """

        print()

        for row in range(self.grid_size):
            row_display = []

            for col in range(self.grid_size):
                position = (row, col)

                if position == self.drone_position:
                    symbol = "D"

                elif position == self.destination:
                    symbol = "G"

                elif position in self.obstacles:
                    symbol = "X"

                else:
                    symbol = "."

                row_display.append(symbol)

            print(" ".join(row_display))

        print()