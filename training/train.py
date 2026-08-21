import numpy as np

from environment.drone_env import (
    DroneEnvironment
)

from agent.q_learning import (
    QLearningAgent
)

from visualization.visualize import (
    visualize_training
)


ACTION_NAMES = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
}


def train_agent(
    episodes=5000,
    max_steps_per_episode=100,
    scenario="standard",
):
    """
    Train dynamic obstacle-aware Q-Learning.

    A new valid obstacle configuration is
    generated for every training episode.

    The agent learns using:
    - position
    - battery level
    - obstacle/boundary information

    Collision is treated as a severe,
    mission-ending failure.
    """

    env = DroneEnvironment(
        scenario=scenario
    )

    agent = QLearningAgent(
        grid_size=env.grid_size,
        max_battery=env.max_battery,
    )

    rewards_per_episode = []
    steps_per_episode = []

    successful_episodes = 0
    battery_failures = 0
    collision_failures = 0

    print(
        "\n==================================="
    )

    print(
        "      DYNAMIC Q-LEARNING"
    )

    print(
        "==================================="
    )

    print(
        f"Scenario: "
        f"{env.scenario_name}"
    )

    print(
        f"Battery Capacity: "
        f"{env.max_battery}"
    )

    print(
        f"Dynamic Obstacles: "
        f"{env.obstacle_count}"
    )

    print(
        f"Episodes: {episodes}"
    )

    print(
        "Obstacle positions change "
        "every episode."
    )

    print(
        "Collision Penalty: -50 "
        "(mission terminating)"
    )

    print(
        "===================================\n"
    )

    for episode in range(
        episodes
    ):

        # --------------------------------------------------
        # Generate a new obstacle configuration
        # --------------------------------------------------

        state = env.reset(
            regenerate_obstacles=True
        )

        total_reward = 0
        steps = 0

        for _ in range(
            max_steps_per_episode
        ):

            action = (
                agent.choose_action(
                    state
                )
            )

            (
                next_state,
                reward,
                done,
            ) = env.step(
                action
            )

            # --------------------------------------------------
            # Learn from transition
            # --------------------------------------------------

            agent.update_q_value(
                state,
                action,
                reward,
                next_state,
                done,
            )

            state = next_state

            total_reward += reward
            steps += 1

            # --------------------------------------------------
            # Terminal-state classification
            # --------------------------------------------------

            if done:

                if (
                    env.drone_position
                    == env.destination
                ):
                    successful_episodes += 1

                elif (
                    env.battery <= 0
                ):
                    battery_failures += 1

                else:
                    # The only remaining terminal
                    # condition is obstacle collision.
                    collision_failures += 1

                break

        # --------------------------------------------------
        # Exploration decay
        # --------------------------------------------------

        agent.decay_epsilon()

        rewards_per_episode.append(
            total_reward
        )

        steps_per_episode.append(
            steps
        )

        # --------------------------------------------------
        # Progress
        # --------------------------------------------------

        if (
            episode + 1
        ) % 100 == 0:

            recent_rewards = (
                rewards_per_episode[
                    -100:
                ]
            )

            recent_steps = (
                steps_per_episode[
                    -100:
                ]
            )

            average_reward = (
                np.mean(
                    recent_rewards
                )
            )

            average_steps = (
                np.mean(
                    recent_steps
                )
            )

            print(
                f"Episode "
                f"{episode + 1}/{episodes} | "
                f"Average Reward: "
                f"{average_reward:.2f} | "
                f"Average Steps: "
                f"{average_steps:.2f} | "
                f"Epsilon: "
                f"{agent.epsilon:.3f}"
            )

    # ==================================================
    # TRAINING STATISTICS
    # ==================================================

    success_rate = (
        successful_episodes
        / episodes
    ) * 100

    battery_failure_rate = (
        battery_failures
        / episodes
    ) * 100

    collision_failure_rate = (
        collision_failures
        / episodes
    ) * 100

    print(
        "\nTraining completed."
    )

    print(
        f"Training Success Rate: "
        f"{success_rate:.2f}%"
    )

    print(
        f"Collision Failure Rate: "
        f"{collision_failure_rate:.2f}%"
    )

    print(
        f"Battery Failure Rate: "
        f"{battery_failure_rate:.2f}%"
    )

    # Store statistics so that the
    # FastAPI backend can expose them.
    env.training_success_rate = (
        success_rate
    )

    env.collision_failure_rate = (
        collision_failure_rate
    )

    env.battery_failure_rate = (
        battery_failure_rate
    )

    return (
        env,
        agent,
        rewards_per_episode,
        steps_per_episode,
    )


def evaluate_agent(
    env,
    agent,
    max_steps=50,
    new_environment=True,
):
    """
    Evaluate Q-Learning using exploitation only.

    By default, a completely new dynamic obstacle
    configuration is generated for evaluation.

    The obstacle configuration remains fixed
    during this single evaluation route.
    """

    state = env.reset(
        regenerate_obstacles=(
            new_environment
        )
    )

    evaluation_obstacles = list(
        env.obstacles
    )

    shortest_possible_steps = (
        env._shortest_path_length(
            env.obstacles
        )
    )

    route = [
        env.drone_position
    ]

    actions_taken = []

    total_reward = 0

    termination_reason = (
        "maximum_steps"
    )

    print(
        "\n==================================="
    )

    print(
        "   DYNAMIC Q-LEARNING EVALUATION"
    )

    print(
        "==================================="
    )

    print(
        f"Scenario: "
        f"{env.scenario_name}"
    )

    print(
        f"Evaluation Obstacles: "
        f"{evaluation_obstacles}"
    )

    print(
        f"Shortest Possible Steps: "
        f"{shortest_possible_steps}"
    )

    for _ in range(
        max_steps
    ):

        state_index = (
            agent._state_index(
                state
            )
        )

        action = int(
            np.argmax(
                agent.q_table[
                    state_index
                ]
            )
        )

        (
            next_state,
            reward,
            done,
        ) = env.step(
            action
        )

        actions_taken.append(
            ACTION_NAMES[
                action
            ]
        )

        route.append(
            env.drone_position
        )

        total_reward += (
            reward
        )

        state = next_state

        if done:

            if (
                env.drone_position
                == env.destination
            ):
                termination_reason = (
                    "successful_delivery"
                )

            elif (
                env.battery <= 0
            ):
                termination_reason = (
                    "battery_depleted"
                )

            else:
                termination_reason = (
                    "obstacle_collision"
                )

            break

    destination_reached = (
        env.drone_position
        == env.destination
    )

    battery_used = (
        env.max_battery
        - env.battery
    )

    actual_steps = len(
        actions_taken
    )

    extra_steps = None

    if (
        destination_reached
        and shortest_possible_steps
        is not None
    ):
        extra_steps = (
            actual_steps
            - shortest_possible_steps
        )

    print(
        "\nRoute:"
    )

    for position in route:
        print(
            position
        )

    print(
        "\nActions:"
    )

    print(
        " -> ".join(
            actions_taken
        )
    )

    print(
        "\nSteps:",
        actual_steps
    )

    print(
        "Shortest Possible Steps:",
        shortest_possible_steps
    )

    print(
        "Extra Steps:",
        extra_steps
    )

    print(
        "Total Reward:",
        total_reward
    )

    print(
        "Battery Remaining:",
        env.battery
    )

    print(
        "Battery Used:",
        battery_used
    )

    print(
        "Destination Reached:",
        destination_reached
    )

    print(
        "Termination Reason:",
        termination_reason
    )

    print(
        "\nFinal Environment:"
    )

    env.render()

    return (
        route,
        actions_taken,
        total_reward,
    )


if __name__ == "__main__":

    selected_scenario = (
        "standard"
    )

    (
        env,
        agent,
        rewards,
        steps,
    ) = train_agent(
        episodes=5000,
        scenario=(
            selected_scenario
        ),
    )

    (
        route,
        actions,
        total_reward,
    ) = evaluate_agent(
        env,
        agent,
    )

    visualize_training(
        env,
        rewards,
        steps,
        route,
    )