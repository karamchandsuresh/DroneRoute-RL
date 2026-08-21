import random
from collections import deque


class DroneEnvironment:
    """
    Dynamic, scenario-based and battery-aware
    environment for DroneRoute RL.

    Obstacles change between episodes.

    The environment guarantees:
    - Start is never blocked.
    - Destination is never blocked.
    - Obstacle count matches the scenario.
    - At least one valid route exists.
    - The generated route is feasible for
      the available battery.

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

    blocked direction:
        0 = free
        1 = obstacle or boundary
    """

    SCENARIOS = {
        "standard": {
            "name": "Standard Delivery",
            "description": (
                "Normal last-mile drone delivery with "
                "dynamically positioned obstacles."
            ),
            "max_battery": 20,
            "obstacle_count": 3,
        },

        "urban": {
            "name": "Urban Restricted-Zone Delivery",
            "description": (
                "Drone delivery through an urban environment "
                "with additional dynamically positioned "
                "buildings and restricted zones."
            ),
            "max_battery": 20,
            "obstacle_count": 5,
        },

        "low_battery": {
            "name": "Low-Battery Delivery",
            "description": (
                "Energy-constrained delivery with dynamically "
                "positioned obstacles and limited battery."
            ),
            "max_battery": 10,
            "obstacle_count": 3,
        },
    }

    def __init__(
        self,
        grid_size=5,
        max_battery=None,
        scenario="standard",
        seed=None,
    ):
        self.grid_size = grid_size

        if scenario not in self.SCENARIOS:
            raise ValueError(
                "Invalid scenario. Choose "
                "'standard', 'urban', or 'low_battery'."
            )

        self.scenario = scenario

        scenario_config = self.SCENARIOS[
            scenario
        ]

        self.scenario_name = (
            scenario_config["name"]
        )

        self.scenario_description = (
            scenario_config["description"]
        )

        self.obstacle_count = (
            scenario_config["obstacle_count"]
        )

        # --------------------------------------------------
        # Start and destination
        # --------------------------------------------------

        self.start_position = (0, 0)

        self.destination = (
            grid_size - 1,
            grid_size - 1,
        )

        # --------------------------------------------------
        # Battery
        # --------------------------------------------------

        if max_battery is None:
            self.max_battery = (
                scenario_config["max_battery"]
            )
        else:
            self.max_battery = max_battery

        # --------------------------------------------------
        # Random generator
        # --------------------------------------------------

        self.random = random.Random(
            seed
        )

        # --------------------------------------------------
        # Actions
        # --------------------------------------------------
        # 0 = UP
        # 1 = DOWN
        # 2 = LEFT
        # 3 = RIGHT

        self.actions = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1),
        }

        # --------------------------------------------------
        # Initial dynamic environment
        # --------------------------------------------------

        self.obstacles = (
            self._generate_valid_obstacles()
        )

        self.drone_position = (
            self.start_position
        )

        self.battery = (
            self.max_battery
        )

    # ==================================================
    # STATE
    # ==================================================

    def get_state(self):
        """
        Return obstacle-aware RL state.
        """

        row, col = (
            self.drone_position
        )

        blocked_directions = []

        for action in range(4):

            (
                row_change,
                col_change,
            ) = self.actions[action]

            next_position = (
                row + row_change,
                col + col_change,
            )

            blocked = (
                not self._is_inside_grid(
                    next_position
                )
                or next_position
                in self.obstacles
            )

            blocked_directions.append(
                int(blocked)
            )

        return (
            row,
            col,
            self.battery,
            blocked_directions[0],
            blocked_directions[1],
            blocked_directions[2],
            blocked_directions[3],
        )

    # ==================================================
    # SCENARIO INFORMATION
    # ==================================================

    def get_scenario_info(self):
        """
        Return information about the
        current generated environment.
        """

        shortest_path = (
            self._shortest_path_length(
                self.obstacles
            )
        )

        return {
            "scenario": self.scenario,
            "name": self.scenario_name,
            "description": (
                self.scenario_description
            ),
            "grid_size": self.grid_size,
            "start": list(
                self.start_position
            ),
            "destination": list(
                self.destination
            ),
            "obstacle_count": (
                self.obstacle_count
            ),
            "obstacles": [
                list(obstacle)
                for obstacle
                in self.obstacles
            ],
            "max_battery": (
                self.max_battery
            ),
            "dynamic_obstacles": True,
            "shortest_possible_steps": (
                shortest_path
            ),
        }

    # ==================================================
    # RESET
    # ==================================================

    def reset(
        self,
        regenerate_obstacles=True,
    ):
        """
        Reset the environment.

        During training:
            regenerate_obstacles=True

        During one evaluation route:
            regenerate_obstacles=False
        """

        if regenerate_obstacles:

            self.obstacles = (
                self._generate_valid_obstacles()
            )

        self.drone_position = (
            self.start_position
        )

        self.battery = (
            self.max_battery
        )

        return self.get_state()

    # ==================================================
    # DYNAMIC OBSTACLE GENERATION
    # ==================================================

    def _generate_valid_obstacles(
        self,
        max_attempts=2000,
    ):
        """
        Generate a random obstacle layout.

        A layout is accepted only when:
        - a route exists,
        - and the route can be completed with
          the available battery.
        """

        available_cells = []

        for row in range(
            self.grid_size
        ):

            for col in range(
                self.grid_size
            ):

                position = (
                    row,
                    col,
                )

                if (
                    position
                    != self.start_position
                    and position
                    != self.destination
                ):

                    available_cells.append(
                        position
                    )

        if (
            self.obstacle_count
            > len(available_cells)
        ):

            raise ValueError(
                "Obstacle count exceeds "
                "available cells."
            )

        for _ in range(
            max_attempts
        ):

            candidate_obstacles = (
                self.random.sample(
                    available_cells,
                    self.obstacle_count,
                )
            )

            shortest_steps = (
                self._shortest_path_length(
                    candidate_obstacles
                )
            )

            # No valid path
            if shortest_steps is None:
                continue

            # Every attempted action consumes one
            # battery unit. Make sure the shortest
            # feasible route fits within battery.
            if (
                shortest_steps
                < self.max_battery
            ):

                return (
                    candidate_obstacles
                )

        raise RuntimeError(
            "Unable to generate a valid "
            "battery-feasible obstacle layout."
        )

    # ==================================================
    # SHORTEST PATH VALIDATION
    # ==================================================

    def _shortest_path_length(
        self,
        obstacles,
    ):
        """
        Return the shortest route length
        using BFS.

        BFS is ONLY used to validate the
        randomly generated environment.

        The RL agent does NOT use BFS
        to select its route.
        """

        blocked = set(
            obstacles
        )

        queue = deque(
            [
                (
                    self.start_position,
                    0,
                )
            ]
        )

        visited = {
            self.start_position
        }

        while queue:

            (
                current_position,
                distance,
            ) = queue.popleft()

            if (
                current_position
                == self.destination
            ):

                return distance

            (
                current_row,
                current_col,
            ) = current_position

            for (
                row_change,
                col_change,
            ) in self.actions.values():

                next_position = (
                    current_row
                    + row_change,
                    current_col
                    + col_change,
                )

                if (
                    self._is_inside_grid(
                        next_position
                    )
                    and next_position
                    not in blocked
                    and next_position
                    not in visited
                ):

                    visited.add(
                        next_position
                    )

                    queue.append(
                        (
                            next_position,
                            distance + 1,
                        )
                    )

        return None

    def _path_exists(
        self,
        obstacles,
    ):
        """
        Return True if at least one path exists.
        """

        return (
            self._shortest_path_length(
                obstacles
            )
            is not None
        )

    # ==================================================
    # STEP
    # ==================================================

    def step(
        self,
        action,
    ):
        """
        Execute one action.

        Rewards:

        Normal movement       = -1
        Boundary violation    = -10
        Obstacle collision    = -50
        Battery depletion     = -50
        Successful delivery   = +100

        A physical obstacle collision is treated
        as a mission-ending failure because a
        real drone could crash or be damaged.
        """

        if action not in self.actions:

            raise ValueError(
                "Invalid action. Choose "
                "0, 1, 2, or 3."
            )

        # Every attempted action uses energy
        self.battery -= 1

        (
            row_change,
            col_change,
        ) = self.actions[action]

        (
            current_row,
            current_col,
        ) = self.drone_position

        new_position = (
            current_row
            + row_change,
            current_col
            + col_change,
        )

        # --------------------------------------------------
        # Battery depleted
        # --------------------------------------------------

        if self.battery <= 0:

            reward = -50
            done = True

            return (
                self.get_state(),
                reward,
                done,
            )

        # --------------------------------------------------
        # Boundary
        # --------------------------------------------------

        if not self._is_inside_grid(
            new_position
        ):

            reward = -10
            done = False

            return (
                self.get_state(),
                reward,
                done,
            )

        # --------------------------------------------------
        # Obstacle collision
        # --------------------------------------------------
        #
        # Collision is treated as a severe,
        # mission-ending event. The drone does
        # not move into the obstacle cell.
        # --------------------------------------------------

        if (
            new_position
            in self.obstacles
        ):

            reward = -50
            done = True

            return (
                self.get_state(),
                reward,
                done,
            )

        # --------------------------------------------------
        # Valid movement
        # --------------------------------------------------

        self.drone_position = (
            new_position
        )

        # --------------------------------------------------
        # Goal
        # --------------------------------------------------

        if (
            self.drone_position
            == self.destination
        ):

            reward = 100
            done = True

            return (
                self.get_state(),
                reward,
                done,
            )

        reward = -1
        done = False

        return (
            self.get_state(),
            reward,
            done,
        )

    # ==================================================
    # GRID VALIDATION
    # ==================================================

    def _is_inside_grid(
        self,
        position,
    ):
        """
        Check whether a position is
        inside the grid.
        """

        row, col = position

        return (
            0
            <= row
            < self.grid_size
            and 0
            <= col
            < self.grid_size
        )

    # ==================================================
    # RENDER
    # ==================================================

    def render(self):
        """
        Print the active dynamic map.
        """

        print()

        print(
            f"Scenario: "
            f"{self.scenario_name}"
        )

        print(
            f"Dynamic Obstacles: "
            f"{self.obstacles}"
        )

        print()

        for row in range(
            self.grid_size
        ):

            row_display = []

            for col in range(
                self.grid_size
            ):

                position = (
                    row,
                    col,
                )

                if (
                    position
                    == self.drone_position
                ):

                    symbol = "D"

                elif (
                    position
                    == self.destination
                ):

                    symbol = "G"

                elif (
                    position
                    in self.obstacles
                ):

                    symbol = "X"

                else:

                    symbol = "."

                row_display.append(
                    symbol
                )

            print(
                " ".join(
                    row_display
                )
            )

        print(
            f"\nBattery: "
            f"{self.battery}/"
            f"{self.max_battery}"
        )

        print()