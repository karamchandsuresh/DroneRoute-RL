import numpy as np

from training.train import train_agent, evaluate_agent
from training.train_dqn import train_dqn, evaluate_dqn


def compare_agents():
    print("=== Q-Learning vs DQN Comparison ===")

    # -----------------------------
    # Q-Learning
    # -----------------------------
    print("\nTraining Q-Learning Agent...")

    q_env, q_agent, q_rewards, q_steps = train_agent()

    q_route, q_actions, q_total_reward = evaluate_agent(
        q_env,
        q_agent
    )

    q_success = (
        q_env.drone_position
        == q_env.destination
    )

    # -----------------------------
    # DQN
    # -----------------------------
    print("\nTraining DQN Agent...")

    (
        dqn_env,
        dqn_agent,
        dqn_rewards,
        dqn_steps,
        dqn_losses
    ) = train_dqn()

    (
        dqn_route,
        dqn_actions,
        dqn_total_reward,
        dqn_success
    ) = evaluate_dqn(
        dqn_env,
        dqn_agent
    )

    # -----------------------------
    # Final metrics
    # -----------------------------
    q_last_100_reward = np.mean(
        q_rewards[-100:]
    )

    dqn_last_100_reward = np.mean(
        dqn_rewards[-100:]
    )

    q_last_100_steps = np.mean(
        q_steps[-100:]
    )

    dqn_last_100_steps = np.mean(
        dqn_steps[-100:]
    )

    print("\n===================================")
    print("      FINAL AGENT COMPARISON")
    print("===================================")

    print("\nQ-Learning")
    print("-----------------------------")
    print(
        "Evaluation Success:",
        q_success
    )
    print(
        "Evaluation Steps:",
        len(q_actions)
    )
    print(
        "Evaluation Reward:",
        q_total_reward
    )
    print(
        "Last 100 Avg Reward:",
        round(q_last_100_reward, 2)
    )
    print(
        "Last 100 Avg Steps:",
        round(q_last_100_steps, 2)
    )

    print("\nDQN")
    print("-----------------------------")
    print(
        "Evaluation Success:",
        dqn_success
    )
    print(
        "Evaluation Steps:",
        len(dqn_actions)
    )
    print(
        "Evaluation Reward:",
        dqn_total_reward
    )
    print(
        "Last 100 Avg Reward:",
        round(
            dqn_last_100_reward,
            2
        )
    )
    print(
        "Last 100 Avg Steps:",
        round(
            dqn_last_100_steps,
            2
        )
    )

    print("\n===================================")
    print("Interpretation")
    print("===================================")

    if (
        len(q_actions)
        == len(dqn_actions)
        and q_total_reward
        == dqn_total_reward
    ):
        print(
            "Both agents learned an equally "
            "efficient final route."
        )

    elif len(q_actions) < len(dqn_actions):
        print(
            "Q-Learning found the shorter "
            "evaluation route."
        )

    else:
        print(
            "DQN found the shorter "
            "evaluation route."
        )

    print(
        "\nQ-Learning is simpler and highly "
        "effective for this small discrete state space."
    )

    print(
        "DQN replaces the Q-table with a neural "
        "network and is better suited to larger "
        "or more complex state spaces."
    )


if __name__ == "__main__":
    compare_agents()