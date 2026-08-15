class DroneEnvironment:
    """
    Battery-aware grid environment for DroneRoute RL.

    The drone must reach the delivery destination while:
    - avoiding obstacles,
    - staying inside the grid,
    - minimizing unnecessary movement,
    - and completing the delivery before its battery is depleted.
    """

    def __init__(
        self,
        grid_size=5,
        max_battery=20
    ):
        self.grid_size = grid_size
        self.max_battery = max_battery

        # Start and destination
        self.start_position = (0, 0)
        self.destination = (
            grid_size - 1,
            grid_size - 1
        )

        # Fixed obstacles / no-fly cells
        self.obstacles = [
            (1, 1),
            (2, 2),
            (3, 1)
        ]

        # Current environment state
        self.drone_position = self.start_position
        self.battery = self.max_battery

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

    def get_state(self):
        """
        Return the current RL state.

        State:
            (row, column, battery)
        """

        row, col = self.drone_position

        return (
            row,
            col,
            self.battery
        )

    def reset(self):
        """
        Reset the environment for a new episode.
        """

        self.drone_position = self.start_position
        self.battery = self.max_battery

        return self.get_state()

    def step(self, action):
        """
        Execute one action in the environment.

        Every attempted action consumes one battery unit.

        Returns:
            next_state:
                (row, column, battery)

            reward:
                Reward or penalty received.

            done:
                True when the episode has finished.
        """

        if action not in self.actions:
            raise ValueError(
                "Invalid action. Choose 0, 1, 2, or 3."
            )

        # Every attempted movement consumes energy
        self.battery -= 1

        row_change, col_change = self.actions[action]

        current_row, current_col = self.drone_position

        new_row = current_row + row_change
        new_col = current_col + col_change

        new_position = (
            new_row,
            new_col
        )

        # --------------------------------------------------
        # Case 1: Battery depleted
        # --------------------------------------------------
        if self.battery <= 0:
            reward = -50
            done = True

            return (
                self.get_state(),
                reward,
                done
            )

        # --------------------------------------------------
        # Case 2: Drone tries to leave the grid
        # --------------------------------------------------
        if not self._is_inside_grid(new_position):
            reward = -10
            done = False

            return (
                self.get_state(),
                reward,
                done
            )

        # --------------------------------------------------
        # Case 3: Drone hits an obstacle
        # --------------------------------------------------
        if new_position in self.obstacles:
            reward = -20
            done = False

            return (
                self.get_state(),
                reward,
                done
            )

        # Valid movement
        self.drone_position = new_position

        # --------------------------------------------------
        # Case 4: Destination reached
        # --------------------------------------------------
        if self.drone_position == self.destination:
            reward = 100
            done = True

            return (
                self.get_state(),
                reward,
                done
            )

        # --------------------------------------------------
        # Case 5: Normal movement
        # --------------------------------------------------
        reward = -1
        done = False

        return (
            self.get_state(),
            reward,
            done
        )

    def _is_inside_grid(self, position):
        """
        Check whether a position lies inside the grid.
        """

        row, col = position

        return (
            0 <= row < self.grid_size
            and 0 <= col < self.grid_size
        )

    def render(self):
        """
        Display the current grid and battery level.
        """

        print()

        for row in range(self.grid_size):

            row_display = []

            for col in range(self.grid_size):

                position = (
                    row,
                    col
                )

                if position == self.drone_position:
                    symbol = "D"

                elif position == self.destination:
                    symbol = "G"

                elif position in self.obstacles:
                    symbol = "X"

                else:
                    symbol = "."

                row_display.append(symbol)

            print(
                " ".join(row_display)
            )

        print(
            f"\nBattery: "
            f"{self.battery}/{self.max_battery}"
        )

        print()