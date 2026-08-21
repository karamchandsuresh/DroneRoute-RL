from collections import deque
from typing import Literal

import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from environment.drone_env import DroneEnvironment
from training.train import train_agent, evaluate_agent
from training.train_dqn import train_dqn, evaluate_dqn


# =========================================================
# FastAPI Application
# =========================================================

app = FastAPI(
    title="DroneRoute RL API",
    description=(
        "Backend API for dynamic obstacle-aware drone "
        "delivery optimization using Q-Learning and "
        "Deep Q-Networks."
    ),
    version="3.1.0",
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://drone-route-rl.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# Scenario Type
# =========================================================

ScenarioType = Literal[
    "standard",
    "urban",
    "low_battery",
]


# =========================================================
# Training Configuration
# =========================================================

Q_LEARNING_EPISODES = {
    "standard": 3000,
    "urban": 3000,
    "low_battery": 3000,
}


DQN_EPISODES = {
    "standard": 1500,
    "urban": 1800,
    "low_battery": 2000,
}


# =========================================================
# Helper Functions
# =========================================================

def shortest_path_length(env):
    """
    Return the shortest possible route length
    for the current obstacle configuration.

    BFS is used only for validation/comparison.
    The RL agents do not use BFS for navigation.
    """

    return env._shortest_path_length(
        env.obstacles
    )


def route_efficiency(
    actual_steps,
    optimal_steps,
):
    """
    Calculate route efficiency.

    100% means the RL route uses the same
    number of steps as the shortest valid route.
    """

    if (
        optimal_steps is None
        or actual_steps <= 0
    ):
        return None

    efficiency = (
        optimal_steps
        / actual_steps
    ) * 100

    return round(
        min(
            efficiency,
            100.0,
        ),
        2,
    )


def find_path_actions(
    env,
    start,
    destination,
):
    """
    Find one safe path between two cells.

    Used only by the educational reward demo.
    It is NOT used by Q-Learning or DQN.
    """

    queue = deque(
        [
            (
                start,
                [],
            )
        ]
    )

    visited = {
        start
    }

    while queue:

        position, actions = (
            queue.popleft()
        )

        if (
            position
            == destination
        ):
            return actions

        row, col = position

        for (
            action,
            (
                row_change,
                col_change,
            ),
        ) in env.actions.items():

            next_position = (
                row + row_change,
                col + col_change,
            )

            if (
                env._is_inside_grid(
                    next_position
                )
                and next_position
                not in env.obstacles
                and next_position
                not in visited
            ):

                visited.add(
                    next_position
                )

                queue.append(
                    (
                        next_position,
                        actions + [action],
                    )
                )

    return []


def find_obstacle_collision_demo(env):
    """
    Find a reachable free cell located next
    to one of the dynamically generated obstacles.

    Returns:
        (
            free_neighbor,
            collision_action
        )

    The collision action points from the
    free cell into the obstacle.
    """

    for obstacle in env.obstacles:

        obstacle_row, obstacle_col = (
            obstacle
        )

        for (
            action,
            (
                row_change,
                col_change,
            ),
        ) in env.actions.items():

            # If performing this action from
            # neighbor_position, drone enters obstacle.
            neighbor_position = (
                obstacle_row
                - row_change,
                obstacle_col
                - col_change,
            )

            if (
                not env._is_inside_grid(
                    neighbor_position
                )
            ):
                continue

            if (
                neighbor_position
                in env.obstacles
            ):
                continue

            path = find_path_actions(
                env,
                env.start_position,
                neighbor_position,
            )

            if (
                neighbor_position
                == env.start_position
                or path
            ):
                return (
                    neighbor_position,
                    action,
                )

    return None


def build_demo_event(
    event_type,
    step,
    action_name,
    from_position,
    attempted_position,
    to_position,
    reward,
    cumulative_reward,
    battery_remaining,
    destination_reached,
    message,
):
    """
    Build one exploration-demo event.
    """

    return {
        "type": event_type,
        "step": step,
        "action": action_name,
        "from": list(
            from_position
        ),
        "attempted": list(
            attempted_position
        ),
        "to": list(
            to_position
        ),
        "reward": reward,
        "cumulative_reward": (
            cumulative_reward
        ),
        "battery_remaining": (
            battery_remaining
        ),
        "destination_reached": (
            destination_reached
        ),
        "message": message,
    }


# =========================================================
# Health Check
# =========================================================

@app.get("/")
def root():
    """
    Basic API health check.
    """

    return {
        "project": "DroneRoute RL",
        "message": (
            "DroneRoute RL API is running"
        ),
        "status": "success",
        "version": "3.1.0",
        "dynamic_obstacles": True,
    }


# =========================================================
# Available Scenarios
# =========================================================

@app.get("/scenarios")
def get_scenarios():
    """
    Return available delivery scenarios.
    """

    scenarios = []

    for scenario_key in (
        DroneEnvironment.SCENARIOS
    ):

        env = DroneEnvironment(
            scenario=scenario_key
        )

        scenarios.append(
            env.get_scenario_info()
        )

    return {
        "dynamic_obstacles": True,
        "scenarios": scenarios,
    }


# =========================================================
# Environment Information
# =========================================================

@app.get("/environment")
def get_environment(
    scenario: ScenarioType = "standard",
):
    """
    Generate a fresh valid dynamic environment.
    """

    env = DroneEnvironment(
        scenario=scenario
    )

    info = (
        env.get_scenario_info()
    )

    info["message"] = (
        "A new valid obstacle configuration "
        "was generated."
    )

    return info


# =========================================================
# Q-Learning Route
# =========================================================

@app.get("/route/q-learning")
def get_q_learning_route(
    scenario: ScenarioType = "standard",
):
    """
    Train Q-Learning across changing obstacle
    configurations and evaluate on a newly
    generated environment.
    """

    episodes = (
        Q_LEARNING_EPISODES[
            scenario
        ]
    )

    (
        env,
        agent,
        rewards,
        steps,
    ) = train_agent(
        episodes=episodes,
        scenario=scenario,
    )

    (
        route,
        actions,
        total_reward,
    ) = evaluate_agent(
        env,
        agent,
        new_environment=True,
    )

    destination_reached = (
        env.drone_position
        == env.destination
    )

    battery_used = (
        env.max_battery
        - env.battery
    )

    optimal_steps = (
        shortest_path_length(
            env
        )
    )

    actual_steps = len(
        actions
    )

    extra_steps = None

    if (
        destination_reached
        and optimal_steps
        is not None
    ):

        extra_steps = (
            actual_steps
            - optimal_steps
        )

    average_reward_last_100 = round(
        float(
            np.mean(
                rewards[-100:]
            )
        ),
        2,
    )

    average_steps_last_100 = round(
        float(
            np.mean(
                steps[-100:]
            )
        ),
        2,
    )

    return {
        "algorithm": "Q-Learning",

        "scenario": scenario,

        "scenario_name": (
            env.scenario_name
        ),

        "scenario_description": (
            env.scenario_description
        ),

        "dynamic_obstacles": True,

        "episodes": episodes,

        "grid_size": (
            env.grid_size
        ),

        "start": list(
            env.start_position
        ),

        "destination": list(
            env.destination
        ),

        "obstacles": [
            list(obstacle)
            for obstacle
            in env.obstacles
        ],

        "route": [
            list(position)
            for position in route
        ],

        "actions": actions,

        "steps": (
            actual_steps
        ),

        "shortest_possible_steps": (
            optimal_steps
        ),

        "extra_steps": (
            extra_steps
        ),

        "route_efficiency": (
            route_efficiency(
                actual_steps,
                optimal_steps,
            )
            if destination_reached
            else None
        ),

        "total_reward": (
            total_reward
        ),

        "battery_capacity": (
            env.max_battery
        ),

        "battery_remaining": (
            env.battery
        ),

        "battery_used": (
            battery_used
        ),

        "destination_reached": (
            destination_reached
        ),

        "training_success_rate": round(
            env.training_success_rate,
            2,
        ),

        "battery_failure_rate": round(
            env.battery_failure_rate,
            2,
        ),

        "average_reward_last_100": (
            average_reward_last_100
        ),

        "average_steps_last_100": (
            average_steps_last_100
        ),
    }


# =========================================================
# DQN Route
# =========================================================

@app.get("/route/dqn")
def get_dqn_route(
    scenario: ScenarioType = "standard",
):
    """
    Train DQN across changing obstacle
    configurations and evaluate on a newly
    generated environment.
    """

    episodes = (
        DQN_EPISODES[
            scenario
        ]
    )

    (
        env,
        agent,
        rewards,
        steps,
        losses,
    ) = train_dqn(
        episodes=episodes,
        scenario=scenario,
    )

    (
        route,
        actions,
        total_reward,
        destination_reached,
    ) = evaluate_dqn(
        env,
        agent,
        new_environment=True,
    )

    battery_used = (
        env.max_battery
        - env.battery
    )

    optimal_steps = (
        shortest_path_length(
            env
        )
    )

    actual_steps = len(
        actions
    )

    extra_steps = None

    if (
        destination_reached
        and optimal_steps
        is not None
    ):

        extra_steps = (
            actual_steps
            - optimal_steps
        )

    average_reward_last_100 = round(
        float(
            np.mean(
                rewards[-100:]
            )
        ),
        2,
    )

    average_steps_last_100 = round(
        float(
            np.mean(
                steps[-100:]
            )
        ),
        2,
    )

    return {
        "algorithm": "DQN",

        "scenario": scenario,

        "scenario_name": (
            env.scenario_name
        ),

        "scenario_description": (
            env.scenario_description
        ),

        "dynamic_obstacles": True,

        "episodes": episodes,

        "grid_size": (
            env.grid_size
        ),

        "start": list(
            env.start_position
        ),

        "destination": list(
            env.destination
        ),

        "obstacles": [
            list(obstacle)
            for obstacle
            in env.obstacles
        ],

        "route": [
            list(position)
            for position in route
        ],

        "actions": actions,

        "steps": (
            actual_steps
        ),

        "shortest_possible_steps": (
            optimal_steps
        ),

        "extra_steps": (
            extra_steps
        ),

        "route_efficiency": (
            route_efficiency(
                actual_steps,
                optimal_steps,
            )
            if destination_reached
            else None
        ),

        "total_reward": (
            total_reward
        ),

        "battery_capacity": (
            env.max_battery
        ),

        "battery_remaining": (
            env.battery
        ),

        "battery_used": (
            battery_used
        ),

        "destination_reached": (
            destination_reached
        ),

        "training_success_rate": round(
            env.training_success_rate,
            2,
        ),

        "battery_failure_rate": round(
            env.battery_failure_rate,
            2,
        ),

        "average_reward_last_100": (
            average_reward_last_100
        ),

        "average_steps_last_100": (
            average_steps_last_100
        ),

        "final_training_loss": (
            round(
                float(
                    env.final_training_loss
                ),
                4,
            )
            if (
                env.final_training_loss
                is not None
            )
            else None
        ),
    }


# =========================================================
# Reward / Penalty Exploration Demo
# =========================================================

@app.get("/demo/exploration")
def exploration_demo():
    """
    Educational demonstration of the reward
    system in a dynamically generated map.

    This endpoint does NOT train Q-Learning
    or DQN.

    It demonstrates four independent examples:

    1. Normal movement     = -1
    2. Boundary violation  = -10
    3. Obstacle collision  = -50 and mission ends
    4. Successful delivery = +100 and mission ends
    """

    env = DroneEnvironment(
        scenario="standard"
    )

    # Keep exactly one dynamic layout
    # for the complete demonstration.
    env.reset(
        regenerate_obstacles=False
    )

    demo_obstacles = list(
        env.obstacles
    )

    action_names = {
        0: "UP",
        1: "DOWN",
        2: "LEFT",
        3: "RIGHT",
    }

    events = []

    cumulative_reward = 0
    step_number = 0


    # =====================================================
    # DEMO 1 — NORMAL MOVEMENT
    # =====================================================

    env.reset(
        regenerate_obstacles=False
    )

    safe_actions = (
        find_path_actions(
            env,
            env.start_position,
            env.destination,
        )
    )

    if safe_actions:

        action = (
            safe_actions[0]
        )

        state_before = (
            env.drone_position
        )

        (
            row_change,
            col_change,
        ) = env.actions[
            action
        ]

        attempted_position = (
            state_before[0]
            + row_change,
            state_before[1]
            + col_change,
        )

        (
            next_state,
            reward,
            done,
        ) = env.step(
            action
        )

        step_number += 1

        cumulative_reward += (
            reward
        )

        events.append(
            build_demo_event(
                event_type="normal",
                step=step_number,
                action_name=(
                    action_names[
                        action
                    ]
                ),
                from_position=(
                    state_before
                ),
                attempted_position=(
                    attempted_position
                ),
                to_position=(
                    env.drone_position
                ),
                reward=reward,
                cumulative_reward=(
                    cumulative_reward
                ),
                battery_remaining=(
                    env.battery
                ),
                destination_reached=False,
                message=(
                    "Valid movement. A small -1 "
                    "cost represents energy and "
                    "time used during navigation."
                ),
            )
        )


    # =====================================================
    # DEMO 2 — BOUNDARY VIOLATION
    # =====================================================

    env.reset(
        regenerate_obstacles=False
    )

    boundary_action = 0  # UP from (0,0)

    state_before = (
        env.drone_position
    )

    (
        row_change,
        col_change,
    ) = env.actions[
        boundary_action
    ]

    attempted_position = (
        state_before[0]
        + row_change,
        state_before[1]
        + col_change,
    )

    (
        next_state,
        reward,
        done,
    ) = env.step(
        boundary_action
    )

    step_number += 1

    cumulative_reward += (
        reward
    )

    events.append(
        build_demo_event(
            event_type="boundary",
            step=step_number,
            action_name=(
                action_names[
                    boundary_action
                ]
            ),
            from_position=(
                state_before
            ),
            attempted_position=(
                attempted_position
            ),
            to_position=(
                env.drone_position
            ),
            reward=reward,
            cumulative_reward=(
                cumulative_reward
            ),
            battery_remaining=(
                env.battery
            ),
            destination_reached=False,
            message=(
                "Boundary violation. The drone "
                "attempted to leave the permitted "
                "operating area and receives -10."
            ),
        )
    )


    # =====================================================
    # DEMO 3 — DYNAMIC OBSTACLE COLLISION
    # =====================================================

    collision_demo = (
        find_obstacle_collision_demo(
            env
        )
    )

    if collision_demo:

        (
            neighbor_position,
            collision_action,
        ) = collision_demo

        # This is an educational example.
        # Place the drone safely beside one of
        # the randomly generated obstacles.
        env.drone_position = (
            neighbor_position
        )

        env.battery = (
            env.max_battery
        )

        state_before = (
            env.drone_position
        )

        (
            row_change,
            col_change,
        ) = env.actions[
            collision_action
        ]

        attempted_position = (
            state_before[0]
            + row_change,
            state_before[1]
            + col_change,
        )

        (
            next_state,
            reward,
            done,
        ) = env.step(
            collision_action
        )

        step_number += 1

        cumulative_reward += (
            reward
        )

        events.append(
            build_demo_event(
                event_type="obstacle",
                step=step_number,
                action_name=(
                    action_names[
                        collision_action
                    ]
                ),
                from_position=(
                    state_before
                ),
                attempted_position=(
                    attempted_position
                ),
                to_position=(
                    env.drone_position
                ),
                reward=reward,
                cumulative_reward=(
                    cumulative_reward
                ),
                battery_remaining=(
                    env.battery
                ),
                destination_reached=False,
                message=(
                    "Obstacle collision. A physical "
                    "collision could damage or crash "
                    "the drone, so the agent receives "
                    "-50 and the mission terminates."
                ),
            )
        )


    # =====================================================
    # DEMO 4 — SUCCESSFUL DELIVERY
    # =====================================================

    env.reset(
        regenerate_obstacles=False
    )

    goal_actions = (
        find_path_actions(
            env,
            env.start_position,
            env.destination,
        )
    )

    if goal_actions:

        # Move safely until one step before goal.
        for action in (
            goal_actions[:-1]
        ):

            (
                next_state,
                reward,
                done,
            ) = env.step(
                action
            )

            if done:
                break

        if (
            env.drone_position
            != env.destination
            and not done
        ):

            final_action = (
                goal_actions[-1]
            )

            state_before = (
                env.drone_position
            )

            (
                row_change,
                col_change,
            ) = env.actions[
                final_action
            ]

            attempted_position = (
                state_before[0]
                + row_change,
                state_before[1]
                + col_change,
            )

            (
                next_state,
                reward,
                done,
            ) = env.step(
                final_action
            )

            step_number += 1

            cumulative_reward += (
                reward
            )

            events.append(
                build_demo_event(
                    event_type="goal",
                    step=step_number,
                    action_name=(
                        action_names[
                            final_action
                        ]
                    ),
                    from_position=(
                        state_before
                    ),
                    attempted_position=(
                        attempted_position
                    ),
                    to_position=(
                        env.drone_position
                    ),
                    reward=reward,
                    cumulative_reward=(
                        cumulative_reward
                    ),
                    battery_remaining=(
                        env.battery
                    ),
                    destination_reached=(
                        env.drone_position
                        == env.destination
                    ),
                    message=(
                        "Destination reached. "
                        "Successful delivery receives "
                        "the maximum +100 reward."
                    ),
                )
            )


    # =====================================================
    # RESPONSE
    # =====================================================

    return {
        "demo": (
            "Dynamic RL Reward "
            "and Safety Demonstration"
        ),

        "scenario": "standard",

        "dynamic_obstacles": True,

        "obstacles": [
            list(obstacle)
            for obstacle
            in demo_obstacles
        ],

        "reward_system": {
            "normal_movement": -1,
            "boundary_violation": -10,
            "obstacle_collision": -50,
            "battery_depletion": -50,
            "successful_delivery": 100,
        },

        "collision_terminates_episode": True,

        "events": events,

        "final_cumulative_reward": (
            cumulative_reward
        ),
    }