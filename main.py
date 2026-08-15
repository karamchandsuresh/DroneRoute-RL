from environment.drone_env import DroneEnvironment
from agent.q_learning import QLearningAgent


def main():
    print("=== DroneRoute RL - Q-Learning Test ===")

    # Create environment
    env = DroneEnvironment()

    # Create Q-Learning agent
    agent = QLearningAgent(
        grid_size=env.grid_size
    )

    print("\nQ-table shape:")
    print(agent.q_table.shape)

    print("\nInitial Q-values at state (0, 0):")
    print(agent.q_table[0, 0])

    # Reset environment
    state = env.reset()

    # Agent chooses an action
    action = agent.choose_action(state)

    print("\nCurrent state:", state)
    print("Chosen action:", action)

    # Environment executes the action
    next_state, reward, done = env.step(action)

    print("Next state:", next_state)
    print("Reward:", reward)
    print("Done:", done)

    # Update Q-value
    agent.update_q_value(
        state,
        action,
        reward,
        next_state,
        done
    )

    print("\nUpdated Q-values at state (0, 0):")
    print(agent.q_table[0, 0])

    # Reduce exploration
    print("\nEpsilon before decay:", agent.epsilon)

    agent.decay_epsilon()

    print("Epsilon after decay:", agent.epsilon)


if __name__ == "__main__":
    main()