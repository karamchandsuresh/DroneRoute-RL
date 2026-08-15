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
    episodes=1000,
    max_steps_per_episode=100
):
    """
    Train the Q-Learning agent.
    """

    env = DroneEnvironment()

    agent = QLearningAgent(
        grid_size=env.grid_size
    )

    rewards_per_episode = []
    steps_per_episode = []

    successful_episodes = 0

    for episode in range(episodes):

        state = env.reset()

        total_reward = 0
        steps = 0

        for step in range(max_steps_per_episode):

            action = agent.choose_action(state)

            next_state, reward, done = env.step(action)

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
                successful_episodes += 1
                break

        agent.decay_epsilon()

        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)

        if (episode + 1) % 100 == 0:

            recent_rewards = rewards_per_episode[-100:]

            average_reward = np.mean(
                recent_rewards
            )

            print(
                f"Episode {episode + 1}/{episodes} | "
                f"Average Reward: {average_reward:.2f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    success_rate = (
        successful_episodes / episodes
    ) * 100

    print("\nTraining completed.")

    print(
        f"Training Success Rate: "
        f"{success_rate:.2f}%"
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
    Evaluate the trained agent using exploitation only.
    """

    state = env.reset()

    route = [state]
    actions_taken = []

    total_reward = 0

    print("\n=== Learned Route Evaluation ===")

    for step in range(max_steps):

        row, col = state

        action = int(
            np.argmax(
                agent.q_table[row, col]
            )
        )

        next_state, reward, done = env.step(
            action
        )

        actions_taken.append(
            ACTION_NAMES[action]
        )

        route.append(next_state)

        total_reward += reward

        state = next_state

        if done:
            break

    destination_reached = (
        state == env.destination
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

    env, agent, rewards, steps = train_agent()

    route, actions, total_reward = evaluate_agent(
        env,
        agent
    )

    visualize_training(
        env,
        rewards,
        steps,
        route
    )