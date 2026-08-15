import os

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = "results"


def create_results_directory():
    """
    Create the results directory if it does not already exist.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)


def moving_average(values, window_size=50):
    """
    Calculate a moving average to make training trends easier to see.
    """
    if len(values) < window_size:
        return np.array(values)

    weights = np.ones(window_size) / window_size

    return np.convolve(
        values,
        weights,
        mode="valid"
    )


def plot_rewards(rewards):
    """
    Plot total reward obtained during each training episode.
    """

    create_results_directory()

    plt.figure(figsize=(10, 5))

    plt.plot(
        rewards,
        alpha=0.35,
        label="Episode Reward"
    )

    average_rewards = moving_average(
        rewards,
        window_size=50
    )

    if len(rewards) >= 50:
        x_values = range(
            49,
            49 + len(average_rewards)
        )

        plt.plot(
            x_values,
            average_rewards,
            linewidth=2,
            label="50-Episode Moving Average"
        )

    plt.title("Q-Learning Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(alpha=0.3)

    output_path = os.path.join(
        RESULTS_DIR,
        "reward_curve.png"
    )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Reward graph saved to: {output_path}")


def plot_steps(steps):
    """
    Plot the number of steps used during each training episode.
    """

    create_results_directory()

    plt.figure(figsize=(10, 5))

    plt.plot(
        steps,
        alpha=0.35,
        label="Steps per Episode"
    )

    average_steps = moving_average(
        steps,
        window_size=50
    )

    if len(steps) >= 50:
        x_values = range(
            49,
            49 + len(average_steps)
        )

        plt.plot(
            x_values,
            average_steps,
            linewidth=2,
            label="50-Episode Moving Average"
        )

    plt.title("Steps Required During Training")
    plt.xlabel("Episode")
    plt.ylabel("Number of Steps")
    plt.legend()
    plt.grid(alpha=0.3)

    output_path = os.path.join(
        RESULTS_DIR,
        "steps_curve.png"
    )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Steps graph saved to: {output_path}")


def plot_learned_route(env, route):
    """
    Visualize the final route learned by the Q-Learning agent.
    """

    create_results_directory()

    grid = np.zeros(
        (env.grid_size, env.grid_size)
    )

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.imshow(
        grid,
        cmap="Greys",
        vmin=0,
        vmax=1
    )

    # Draw grid lines
    ax.set_xticks(
        np.arange(-0.5, env.grid_size, 1),
        minor=True
    )

    ax.set_yticks(
        np.arange(-0.5, env.grid_size, 1),
        minor=True
    )

    ax.grid(
        which="minor",
        linewidth=1.5
    )

    # Remove normal tick labels
    ax.set_xticks([])
    ax.set_yticks([])

    # Obstacles
    for row, col in env.obstacles:
        ax.text(
            col,
            row,
            "X",
            ha="center",
            va="center",
            fontsize=22,
            fontweight="bold"
        )

    # Start
    start_row, start_col = env.start_position

    ax.text(
        start_col,
        start_row,
        "START",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold"
    )

    # Goal
    goal_row, goal_col = env.destination

    ax.text(
        goal_col,
        goal_row,
        "GOAL",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold"
    )

    # Convert route coordinates for plotting
    route_rows = [
        position[0]
        for position in route
    ]

    route_cols = [
        position[1]
        for position in route
    ]

    ax.plot(
        route_cols,
        route_rows,
        marker="o",
        linewidth=2.5,
        label="Learned Route"
    )

    ax.set_title(
        "DroneRoute RL - Learned Delivery Route"
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05)
    )

    output_path = os.path.join(
        RESULTS_DIR,
        "learned_route.png"
    )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Route visualization saved to: {output_path}")


def visualize_training(
    env,
    rewards,
    steps,
    route
):
    """
    Generate all project visualizations.
    """

    print("\n=== Generating Visualizations ===")

    plot_rewards(rewards)
    plot_steps(steps)
    plot_learned_route(env, route)

    print("Visualization completed.")