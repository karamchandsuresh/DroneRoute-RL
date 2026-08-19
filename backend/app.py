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
        "Backend API for scenario-based drone delivery "
        "optimization using Q-Learning and Deep Q-Networks."
    ),
    version="2.0.0"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173"
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
    "low_battery"
]


# =========================================================
# Scenario Training Configuration
# =========================================================

Q_LEARNING_EPISODES = {
    "standard": 1500,
    "urban": 2000,
    "low_battery": 2500
}


DQN_EPISODES = {
    "standard": 1000,
    "urban": 1500,
    "low_battery": 2000
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
        "message": "DroneRoute RL API is running",
        "status": "success",
        "version": "2.0.0"
    }


# =========================================================
# Available Scenarios
# =========================================================

@app.get("/scenarios")
def get_scenarios():
    """
    Return all delivery scenarios available
    in the simulator.
    """

    scenarios = []

    for scenario_key in DroneEnvironment.SCENARIOS:

        env = DroneEnvironment(
            scenario=scenario_key
        )

        scenarios.append(
            env.get_scenario_info()
        )

    return {
        "scenarios": scenarios
    }


# =========================================================
# Environment Information
# =========================================================

@app.get("/environment")
def get_environment(
    scenario: ScenarioType = "standard"
):
    """
    Return the environment configuration
    for the selected delivery scenario.
    """

    env = DroneEnvironment(
        scenario=scenario
    )

    return env.get_scenario_info()


# =========================================================
# Q-Learning Route
# =========================================================

@app.get("/route/q-learning")
def get_q_learning_route(
    scenario: ScenarioType = "standard"
):
    """
    Train and evaluate the Q-Learning agent
    for the selected scenario.
    """

    episodes = Q_LEARNING_EPISODES[
        scenario
    ]

    (
        env,
        agent,
        rewards,
        steps
    ) = train_agent(
        episodes=episodes,
        scenario=scenario
    )

    (
        route,
        actions,
        total_reward
    ) = evaluate_agent(
        env,
        agent
    )

    destination_reached = (
        env.drone_position
        == env.destination
    )

    battery_used = (
        env.max_battery
        - env.battery
    )

    average_reward_last_100 = round(
        float(
            np.mean(
                rewards[-100:]
            )
        ),
        2
    )

    average_steps_last_100 = round(
        float(
            np.mean(
                steps[-100:]
            )
        ),
        2
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

        "episodes": episodes,

        "route": [
            list(position)
            for position in route
        ],

        "actions": actions,

        "steps": len(actions),

        "total_reward": total_reward,

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
            2
        ),

        "battery_failure_rate": round(
            env.battery_failure_rate,
            2
        ),

        "average_reward_last_100": (
            average_reward_last_100
        ),

        "average_steps_last_100": (
            average_steps_last_100
        )
    }


# =========================================================
# DQN Route
# =========================================================

@app.get("/route/dqn")
def get_dqn_route(
    scenario: ScenarioType = "standard"
):
    """
    Train and evaluate the DQN agent
    for the selected delivery scenario.
    """

    episodes = DQN_EPISODES[
        scenario
    ]

    (
        env,
        agent,
        rewards,
        steps,
        losses
    ) = train_dqn(
        episodes=episodes,
        scenario=scenario
    )

    (
        route,
        actions,
        total_reward,
        destination_reached
    ) = evaluate_dqn(
        env,
        agent
    )

    battery_used = (
        env.max_battery
        - env.battery
    )

    average_reward_last_100 = round(
        float(
            np.mean(
                rewards[-100:]
            )
        ),
        2
    )

    average_steps_last_100 = round(
        float(
            np.mean(
                steps[-100:]
            )
        ),
        2
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

        "episodes": episodes,

        "route": [
            list(position)
            for position in route
        ],

        "actions": actions,

        "steps": len(actions),

        "total_reward": total_reward,

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
            2
        ),

        "battery_failure_rate": round(
            env.battery_failure_rate,
            2
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
                4
            )
            if env.final_training_loss
            is not None
            else None
        )
    }


# =========================================================
# Reward / Penalty Exploration Demo
# =========================================================

@app.get("/demo/exploration")
def exploration_demo():
    """
    Demonstrate the reward system using one
    continuous journey in the Standard Delivery
    environment.

    This demo is intentionally separate from
    trained Q-Learning and DQN policies.
    """

    env = DroneEnvironment(
        scenario="standard"
    )

    events = []

    cumulative_reward = 0
    step_number = 0

    action_names = {
        0: "UP",
        1: "DOWN",
        2: "LEFT",
        3: "RIGHT"
    }

    # -----------------------------------------------------
    # Demonstration sequence
    # -----------------------------------------------------
    #
    # Start: (0, 0)
    #
    # UP
    #   -> Boundary collision (-10)
    #
    # RIGHT
    #   -> Normal movement (-1)
    #
    # DOWN
    #   -> Obstacle collision at (1,1) (-20)
    #
    # Remaining actions form a safe path
    # toward the destination.
    #
    # Final action
    #   -> Successful delivery (+100)
    # -----------------------------------------------------

    demonstration_actions = [
        0,  # UP - boundary collision
        3,  # RIGHT
        1,  # DOWN - obstacle collision
        3,  # RIGHT
        3,  # RIGHT
        3,  # RIGHT
        1,  # DOWN
        1,  # DOWN
        1,  # DOWN
        1   # DOWN - goal
    ]

    for action in demonstration_actions:

        step_number += 1

        state_before = (
            env.drone_position
        )

        row_change, col_change = (
            env.actions[action]
        )

        attempted_position = (
            state_before[0]
            + row_change,

            state_before[1]
            + col_change
        )

        (
            next_state,
            reward,
            done
        ) = env.step(
            action
        )

        cumulative_reward += reward

        # -------------------------------------------------
        # Determine event type
        # -------------------------------------------------

        if (
            done
            and env.drone_position
            == env.destination
        ):

            event_type = "goal"

            message = (
                "Destination reached. "
                "Large positive reward received."
            )

        elif not env._is_inside_grid(
            attempted_position
        ):

            event_type = "boundary"

            message = (
                "Boundary collision. "
                "Drone cannot leave the "
                "permitted operating area."
            )

        elif (
            attempted_position
            in env.obstacles
        ):

            event_type = "obstacle"

            message = (
                "Obstacle collision. "
                "The drone receives a penalty "
                "and remains in its previous position."
            )

        else:

            event_type = "normal"

            message = (
                "Valid movement. "
                "A small movement cost is applied."
            )

        events.append({
            "type": event_type,

            "step": step_number,

            "action": (
                action_names[action]
            ),

            "from": list(
                state_before
            ),

            "attempted": list(
                attempted_position
            ),

            "to": list(
                env.drone_position
            ),

            "reward": reward,

            "cumulative_reward": (
                cumulative_reward
            ),

            "battery_used": (
                env.max_battery
                - env.battery
            ),

            "battery_remaining": (
                env.battery
            ),

            "destination_reached": (
                env.drone_position
                == env.destination
            ),

            "message": message
        })

        if done:
            break

    return {
        "demo": (
            "RL Exploration and "
            "Reward Demonstration"
        ),

        "scenario": "standard",

        "reward_system": {
            "normal_movement": -1,
            "boundary_violation": -10,
            "obstacle_collision": -20,
            "battery_depletion": -50,
            "successful_delivery": 100
        },

        "events": events,

        "final_cumulative_reward": (
            cumulative_reward
        )
    }