import numpy as np
import torch

from environment.drone_env import DroneEnvironment
from agent.dqn_agent import DQNAgent


ACTION_NAMES = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT"
}


def manhattan_distance(position, destination):
    """
    Calculate Manhattan distance between
    the drone and the destination.

    Example:
        (0, 0) -> (4, 4) = 8
    """

    return (
        abs(position[0] - destination[0])
        + abs(position[1] - destination[1])
    )


def get_training_reward(
    env,
    previous_position,
    new_position,
    environment_reward,
    done
):
    """
    Apply reward shaping during DQN training.

    IMPORTANT:
    This does NOT change the actual environment reward.

    The environment still uses:
        Normal movement      = -1
        Boundary violation   = -10
        Obstacle collision   = -20
        Battery depletion    = -50
        Destination reached  = +100

    Reward shaping only provides additional guidance
    to DQN while it is learning.
    """

    # ----------------------------------------------
    # Preserve terminal rewards
    # ----------------------------------------------

    if done:

        if env.drone_position == env.destination:
            return environment_reward

        # Battery depletion
        return environment_reward

    # ----------------------------------------------
    # Preserve collisions / invalid movements
    # ----------------------------------------------

    if environment_reward <= -10:
        return environment_reward

    # ----------------------------------------------
    # Compare distance to destination
    # ----------------------------------------------

    previous_distance = manhattan_distance(
        previous_position,
        env.destination
    )

    new_distance = manhattan_distance(
        new_position,
        env.destination
    )

    # Moving closer to the destination
    if new_distance < previous_distance:

        return environment_reward + 2.0

    # Moving farther from the destination
    if new_distance > previous_distance:

        return environment_reward - 2.0

    # No useful progress
    return environment_reward


def train_dqn(
    episodes=1000,
    max_steps_per_episode=100,
    target_update_frequency=10,
    scenario="standard"
):
    """
    Train the DQN agent in the selected
    battery-aware delivery scenario.

    Available scenarios:
        standard
        urban
        low_battery

    DQN uses training-only reward shaping to
    provide additional directional feedback.
    """

    # --------------------------------------------------
    # Create selected scenario
    # --------------------------------------------------

    env = DroneEnvironment(
        scenario=scenario
    )

    agent = DQNAgent(
        grid_size=env.grid_size,
        max_battery=env.max_battery
    )

    rewards_per_episode = []
    steps_per_episode = []
    losses = []

    successful_episodes = 0
    battery_failures = 0

    print("\n===================================")
    print("          DQN TRAINING")
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
        f"Device: {agent.device}"
    )

    print(
        f"Episodes: {episodes}"
    )

    print(
        "Reward Shaping: Enabled"
    )

    print("===================================\n")

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    for episode in range(episodes):

        state = env.reset()

        # Actual environment reward is stored here.
        # This keeps our statistics comparable with
        # Q-Learning and final evaluation.
        total_reward = 0

        steps = 0

        for step in range(
            max_steps_per_episode
        ):

            # ------------------------------------------
            # 1. Epsilon-greedy action selection
            # ------------------------------------------

            action = agent.choose_action(
                state
            )

            # Save position before movement
            previous_position = (
                env.drone_position
            )

            # ------------------------------------------
            # 2. Execute action
            # ------------------------------------------

            next_state, reward, done = (
                env.step(action)
            )

            new_position = (
                env.drone_position
            )

            # ------------------------------------------
            # 3. Training-only reward shaping
            # ------------------------------------------

            training_reward = (
                get_training_reward(
                    env=env,
                    previous_position=previous_position,
                    new_position=new_position,
                    environment_reward=reward,
                    done=done
                )
            )

            # ------------------------------------------
            # 4. Store shaped experience
            # ------------------------------------------

            agent.remember(
                state,
                action,
                training_reward,
                next_state,
                done
            )

            # ------------------------------------------
            # 5. Experience replay
            # ------------------------------------------

            loss = agent.replay()

            if loss is not None:
                losses.append(loss)

            state = next_state

            # IMPORTANT:
            # Statistics use the ORIGINAL environment
            # reward, not the shaped training reward.
            total_reward += reward

            steps += 1

            if done:

                if (
                    env.drone_position
                    == env.destination
                ):
                    successful_episodes += 1

                elif env.battery <= 0:
                    battery_failures += 1

                break

        # ----------------------------------------------
        # Reduce exploration
        # ----------------------------------------------

        agent.decay_epsilon()

        # ----------------------------------------------
        # Update target network
        # ----------------------------------------------

        if (
            episode + 1
        ) % target_update_frequency == 0:

            agent.update_target_network()

        rewards_per_episode.append(
            total_reward
        )

        steps_per_episode.append(
            steps
        )

        # ----------------------------------------------
        # Training progress
        # ----------------------------------------------

        if (
            episode + 1
        ) % 100 == 0:

            recent_rewards = (
                rewards_per_episode[-100:]
            )

            average_reward = np.mean(
                recent_rewards
            )

            recent_steps = (
                steps_per_episode[-100:]
            )

            average_steps = np.mean(
                recent_steps
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

    # --------------------------------------------------
    # Training statistics
    # --------------------------------------------------

    success_rate = (
        successful_episodes
        / episodes
    ) * 100

    battery_failure_rate = (
        battery_failures
        / episodes
    ) * 100

    print(
        "\nDQN training completed."
    )

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

    if losses:

        print(
            f"Final Training Loss: "
            f"{losses[-1]:.4f}"
        )

    # Store statistics for API access

    env.training_success_rate = (
        success_rate
    )

    env.battery_failure_rate = (
        battery_failure_rate
    )

    if losses:

        env.final_training_loss = (
            losses[-1]
        )

    else:

        env.final_training_loss = None

    return (
        env,
        agent,
        rewards_per_episode,
        steps_per_episode,
        losses
    )


def evaluate_dqn(
    env,
    agent,
    max_steps=50
):
    """
    Evaluate the trained DQN using
    exploitation only.

    IMPORTANT:
    Reward shaping is NOT used here.

    Evaluation uses the original environment
    reward system.
    """

    state = env.reset()

    route = [
        env.drone_position
    ]

    actions_taken = []

    total_reward = 0

    print(
        "\n=== DQN Route Evaluation ==="
    )

    print(
        f"Scenario: "
        f"{env.scenario_name}"
    )

    for step in range(max_steps):

        # ----------------------------------------------
        # Normalize state
        # ----------------------------------------------

        normalized_state = (
            agent.normalize_state(
                state
            )
        )

        state_tensor = torch.tensor(
            normalized_state,
            dtype=torch.float32,
            device=agent.device
        ).unsqueeze(0)

        # ----------------------------------------------
        # Exploitation only
        # ----------------------------------------------

        with torch.no_grad():

            q_values = (
                agent.policy_network(
                    state_tensor
                )
            )

            action = int(
                torch.argmax(
                    q_values,
                    dim=1
                ).item()
            )

        # ----------------------------------------------
        # Execute action
        # ----------------------------------------------

        next_state, reward, done = (
            env.step(action)
        )

        actions_taken.append(
            ACTION_NAMES[action]
        )

        route.append(
            env.drone_position
        )

        # ORIGINAL environment reward
        total_reward += reward

        state = next_state

        if done:
            break

    # --------------------------------------------------
    # Evaluation statistics
    # --------------------------------------------------

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
        total_reward,
        destination_reached
    )


if __name__ == "__main__":

    # Default scenario when running:
    #
    # python -m training.train_dqn
    #
    # Other options:
    #     urban
    #     low_battery

    selected_scenario = "standard"

    env, agent, rewards, steps, losses = (
        train_dqn(
            scenario=selected_scenario
        )
    )

    (
        route,
        actions,
        total_reward,
        reached
    ) = evaluate_dqn(
        env,
        agent
    )