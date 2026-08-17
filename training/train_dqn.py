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


def train_dqn(
    episodes=1000,
    max_steps_per_episode=100,
    target_update_frequency=10
):
    """
    Train the DQN agent in the battery-aware drone environment.
    """

    env = DroneEnvironment()

    agent = DQNAgent(
        grid_size=env.grid_size,
        max_battery=env.max_battery
    )

    rewards_per_episode = []
    steps_per_episode = []
    losses = []

    successful_episodes = 0
    battery_failures = 0

    print("=== DQN Training ===")
    print("Device:", agent.device)
    print("Episodes:", episodes)

    for episode in range(episodes):

        state = env.reset()

        total_reward = 0
        steps = 0

        for step in range(max_steps_per_episode):

            # 1. Select action using epsilon-greedy strategy
            action = agent.choose_action(state)

            # 2. Execute action in environment
            next_state, reward, done = env.step(action)

            # 3. Store experience in replay memory
            agent.remember(
                state,
                action,
                reward,
                next_state,
                done
            )

            # 4. Learn from a random batch of past experiences
            loss = agent.replay()

            if loss is not None:
                losses.append(loss)

            state = next_state

            total_reward += reward
            steps += 1

            if done:

                if env.drone_position == env.destination:
                    successful_episodes += 1

                elif env.battery <= 0:
                    battery_failures += 1

                break

        # Reduce exploration after each episode
        agent.decay_epsilon()

        # Periodically update target network
        if (episode + 1) % target_update_frequency == 0:
            agent.update_target_network()

        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)

        if (episode + 1) % 100 == 0:

            recent_rewards = rewards_per_episode[-100:]

            average_reward = np.mean(recent_rewards)

            recent_steps = steps_per_episode[-100:]

            average_steps = np.mean(recent_steps)

            print(
                f"Episode {episode + 1}/{episodes} | "
                f"Average Reward: {average_reward:.2f} | "
                f"Average Steps: {average_steps:.2f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    success_rate = (
        successful_episodes / episodes
    ) * 100

    battery_failure_rate = (
        battery_failures / episodes
    ) * 100

    print("\nDQN training completed.")

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
    Evaluate the trained DQN using exploitation only.
    """

    state = env.reset()

    route = [env.drone_position]
    actions_taken = []

    total_reward = 0

    print("\n=== DQN Route Evaluation ===")

    for step in range(max_steps):

        normalized_state = agent.normalize_state(
            state
        )

        state_tensor = torch.tensor(
            normalized_state,
            dtype=torch.float32,
            device=agent.device
        ).unsqueeze(0)

        # No exploration during evaluation
        with torch.no_grad():

            q_values = agent.policy_network(
                state_tensor
            )

            action = int(
                torch.argmax(
                    q_values,
                    dim=1
                ).item()
            )

        next_state, reward, done = env.step(
            action
        )

        actions_taken.append(
            ACTION_NAMES[action]
        )

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

    print("\nRoute:")

    for position in route:
        print(position)

    print("\nActions:")

    print(
        " -> ".join(actions_taken)
    )

    print("\nSteps:", len(actions_taken))
    print("Total Reward:", total_reward)

    print(
        "Battery Remaining:",
        env.battery
    )

    print(
        "Battery Used:",
        env.max_battery - env.battery
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

    env, agent, rewards, steps, losses = train_dqn()

    route, actions, total_reward, reached = evaluate_dqn(
        env,
        agent
    )