import numpy as np

from environment.drone_env import DroneEnvironment
from agent.q_learning import QLearningAgent
from visualization.visualize import visualize_training


ACTION_NAMES = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT"
}


def train_agent(
    episodes=1500,
    max_steps_per_episode=100,
    scenario="standard"
):
    """
    Train the battery-aware Q-Learning agent
    for the selected delivery scenario.

    Available scenarios:
        standard
        urban
        low_battery
    """

    # Create the selected delivery environment
    env = DroneEnvironment(
        scenario=scenario
    )

    print("\n===================================")
    print("      Q-LEARNING TRAINING")
    print("===================================")

    print(
        f"Scenario: {env.scenario_name}"
    )

    print(
        f"Battery Capacity: "
        f"{env.max_battery}"
    )

    print(
        f"Obstacles: "
        f"{len(env.obstacles)}"
    )

    print(
        f"Episodes: {episodes}"
    )

    print("===================================\n")

    agent = QLearningAgent(
        grid_size=env.grid_size,
        max_battery=env.max_battery
    )

    rewards_per_episode = []
    steps_per_episode = []

    successful_episodes = 0
    battery_failures = 0

    for episode in range(episodes):

        state = env.reset()

        total_reward = 0
        steps = 0

        for step in range(
            max_steps_per_episode
        ):

            # ----------------------------------------------
            # Agent chooses an action
            # ----------------------------------------------

            action = agent.choose_action(
                state
            )

            # ----------------------------------------------
            # Environment executes the action
            # ----------------------------------------------

            next_state, reward, done = (
                env.step(action)
            )

            # ----------------------------------------------
            # Agent learns from the experience
            # ----------------------------------------------

            agent.update_q_value(
                state,
                action,
                reward,
                next_state,
                done
            )

            state = next_state

            total_reward += reward
            steps += 1

            if done:

                # Successful delivery
                if (
                    env.drone_position
                    == env.destination
                ):
                    successful_episodes += 1

                # Battery depleted
                elif env.battery <= 0:
                    battery_failures += 1

                break

        # Reduce exploration gradually
        agent.decay_epsilon()

        rewards_per_episode.append(
            total_reward
        )

        steps_per_episode.append(
            steps
        )

        if (
            episode + 1
        ) % 100 == 0:

            recent_rewards = (
                rewards_per_episode[-100:]
            )

            average_reward = np.mean(
                recent_rewards
            )

            print(
                f"Episode "
                f"{episode + 1}/{episodes} | "
                f"Average Reward: "
                f"{average_reward:.2f} | "
                f"Epsilon: "
                f"{agent.epsilon:.3f}"
            )

    # ------------------------------------------------------
    # Training statistics
    # ------------------------------------------------------

    success_rate = (
        successful_episodes
        / episodes
    ) * 100

    battery_failure_rate = (
        battery_failures
        / episodes
    ) * 100

    print("\nTraining completed.")

    print(
        f"Scenario: "
        f"{env.scenario_name}"
    )

    print(
        f"Training Success Rate: "
        f"{success_rate:.2f}%"
    )

    print(
        f"Battery Failure Rate: "
        f"{battery_failure_rate:.2f}%"
    )

    # Store statistics on the environment.
    # This allows the API to retrieve them later
    # without changing the existing return structure.

    env.training_success_rate = (
        success_rate
    )

    env.battery_failure_rate = (
        battery_failure_rate
    )

    return (
        env,
        agent,
        rewards_per_episode,
        steps_per_episode
    )


def evaluate_agent(
    env,
    agent,
    max_steps=50
):
    """
    Evaluate the trained battery-aware
    Q-Learning agent using exploitation only.
    """

    state = env.reset()

    # Store positions for route visualization
    route = [
        env.drone_position
    ]

    actions_taken = []

    total_reward = 0

    print(
        "\n=== Battery-Aware "
        "Route Evaluation ==="
    )

    print(
        f"Scenario: "
        f"{env.scenario_name}"
    )

    for step in range(max_steps):

        row, col, battery = state

        # ----------------------------------------------
        # Exploitation only
        # ----------------------------------------------
        # Select the action with the highest
        # learned Q-value.

        action = int(
            np.argmax(
                agent.q_table[
                    row,
                    col,
                    battery
                ]
            )
        )

        next_state, reward, done = (
            env.step(action)
        )

        actions_taken.append(
            ACTION_NAMES[action]
        )

        # Save only the drone position
        route.append(
            env.drone_position
        )

        total_reward += reward

        state = next_state

        if done:
            break

    destination_reached = (
        env.drone_position
        == env.destination
    )

    battery_used = (
        env.max_battery
        - env.battery
    )

    print("\nRoute:")

    for position in route:
        print(position)

    print("\nActions:")

    print(
        " -> ".join(
            actions_taken
        )
    )

    print(
        "\nSteps:",
        len(actions_taken)
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

    print("\nFinal Environment:")

    env.render()

    return (
        route,
        actions_taken,
        total_reward
    )


if __name__ == "__main__":

    # Running this file directly uses the
    # standard delivery scenario.
    #
    # Other available scenarios:
    #     "urban"
    #     "low_battery"

    selected_scenario = "standard"

    env, agent, rewards, steps = (
        train_agent(
            scenario=selected_scenario
        )
    )

    route, actions, total_reward = (
        evaluate_agent(
            env,
            agent
        )
    )

    visualize_training(
        env,
        rewards,
        steps,
        route
    )