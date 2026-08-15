import numpy as np

from environment.drone_env import DroneEnvironment
from agent.q_learning import QLearningAgent


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

    for episode in range(episodes):

        state = env.reset()

        total_reward = 0
        steps = 0

        for step in range(max_steps_per_episode):

            # Choose action using epsilon-greedy strategy
            action = agent.choose_action(state)

            # Perform action
            next_state, reward, done = env.step(action)

            # Update Q-table
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
                break

        # Reduce exploration
        agent.decay_epsilon()

        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)

        # Print training progress every 100 episodes
        if (episode + 1) % 100 == 0:

            recent_rewards = rewards_per_episode[-100:]

            average_reward = np.mean(recent_rewards)

            print(
                f"Episode {episode + 1}/{episodes} | "
                f"Average Reward: {average_reward:.2f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    print("\nTraining completed.")

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
    Evaluate the trained agent using only exploitation.

    No random exploration is used during evaluation.
    """

    state = env.reset()

    route = [state]
    actions_taken = []

    total_reward = 0

    print("\n=== Learned Route Evaluation ===")

    for step in range(max_steps):

        row, col = state

        # Always choose the action with the highest Q-value
        action = int(
            np.argmax(agent.q_table[row, col])
        )

        next_state, reward, done = env.step(action)

        actions_taken.append(
            ACTION_NAMES[action]
        )

        route.append(next_state)

        total_reward += reward

        state = next_state

        if done:
            break

    print("\nRoute:")

    for position in route:
        print(position)

    print("\nActions:")
    print(" -> ".join(actions_taken))

    print("\nSteps:", len(actions_taken))
    print("Total Reward:", total_reward)
    print("Destination Reached:", state == env.destination)

    print("\nFinal Environment:")
    env.render()

    return route, actions_taken, total_reward


if __name__ == "__main__":

    env, agent, rewards, steps = train_agent()

    evaluate_agent(
        env,
        agent
    )