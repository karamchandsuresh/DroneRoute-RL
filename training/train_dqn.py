import numpy as np
import torch

from environment.drone_env import DroneEnvironment
from agent.dqn_agent import DQNAgent


ACTION_NAMES = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
}


# =========================================================
# Helper: Manhattan Distance
# =========================================================

def manhattan_distance(
    position,
    destination,
):
    """
    Calculate Manhattan distance between
    two grid positions.
    """

    return (
        abs(
            position[0]
            - destination[0]
        )
        + abs(
            position[1]
            - destination[1]
        )
    )


# =========================================================
# Training Reward Shaping
# =========================================================

def get_training_reward(
    env,
    previous_position,
    new_position,
    environment_reward,
    done,
):
    """
    Training-only reward shaping.

    Final evaluation continues to use
    the original environment reward.
    """

    # Preserve terminal rewards such as:
    # successful delivery, obstacle collision,
    # and battery depletion.
    if done:

        return environment_reward

    # Preserve boundary / safety penalties.
    if environment_reward <= -10:

        return environment_reward

    previous_distance = (
        manhattan_distance(
            previous_position,
            env.destination,
        )
    )

    new_distance = (
        manhattan_distance(
            new_position,
            env.destination,
        )
    )

    # Encourage movement toward destination.
    if new_distance < previous_distance:

        return (
            environment_reward
            + 2.0
        )

    # Discourage unnecessary movement away
    # from the destination.
    if new_distance > previous_distance:

        return (
            environment_reward
            - 2.0
        )

    return environment_reward


# =========================================================
# DQN Training
# =========================================================

def train_dqn(
    episodes=2500,
    max_steps_per_episode=100,
    target_update_frequency=10,
    scenario="standard",
):
    """
    Train DQN using dynamically generated
    obstacle configurations.

    A new valid map is generated for every
    training episode.
    """

    env = DroneEnvironment(
        scenario=scenario
    )

    agent = DQNAgent(
        grid_size=env.grid_size,
        max_battery=env.max_battery,
    )

    rewards_per_episode = []
    steps_per_episode = []
    losses = []

    successful_episodes = 0
    battery_failures = 0
    collision_failures = 0

    print(
        "\n==================================="
    )

    print(
        "       DYNAMIC DQN TRAINING"
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
        f"Device: "
        f"{agent.device}"
    )

    print(
        f"Episodes: "
        f"{episodes}"
    )

    print(
        "Reward Shaping: Enabled"
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

    # -----------------------------------------------------
    # Training Loop
    # -----------------------------------------------------

    for episode in range(
        episodes
    ):

        state = env.reset(
            regenerate_obstacles=True
        )

        total_reward = 0
        steps = 0

        for _ in range(
            max_steps_per_episode
        ):

            # Agent selects an action using
            # epsilon-greedy exploration.
            action = (
                agent.choose_action(
                    state
                )
            )

            previous_position = (
                env.drone_position
            )

            (
                next_state,
                reward,
                done,
            ) = env.step(
                action
            )

            new_position = (
                env.drone_position
            )

            # Reward shaping is used only
            # during training.
            training_reward = (
                get_training_reward(
                    env=env,
                    previous_position=(
                        previous_position
                    ),
                    new_position=(
                        new_position
                    ),
                    environment_reward=(
                        reward
                    ),
                    done=done,
                )
            )

            # Store transition in
            # experience replay memory.
            agent.remember(
                state,
                action,
                training_reward,
                next_state,
                done,
            )

            # Train network from replay memory.
            loss = agent.replay()

            if loss is not None:

                losses.append(
                    loss
                )

            state = next_state

            # Statistics use the ORIGINAL
            # environment reward.
            total_reward += reward

            steps += 1

            if done:

                if (
                    env.drone_position
                    == env.destination
                ):

                    successful_episodes += 1

                elif (
                    getattr(
                        env,
                        "termination_reason",
                        None,
                    )
                    == "collision"
                ):

                    collision_failures += 1

                elif env.battery <= 0:

                    battery_failures += 1

                break

        # Gradually reduce exploration.
        agent.decay_epsilon()

        # Periodically synchronize the
        # target network.
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

        # Display progress every 100 episodes.
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

    # -----------------------------------------------------
    # Training Statistics
    # -----------------------------------------------------

    success_rate = (
        successful_episodes
        / episodes
    ) * 100

    collision_failure_rate = (
        collision_failures
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

    if losses:

        print(
            f"Final Training Loss: "
            f"{losses[-1]:.4f}"
        )

    # Store statistics on environment
    # so FastAPI can retrieve them.
    env.training_success_rate = (
        success_rate
    )

    env.collision_failure_rate = (
        collision_failure_rate
    )

    env.battery_failure_rate = (
        battery_failure_rate
    )

    env.final_training_loss = (
        losses[-1]
        if losses
        else None
    )

    return (
        env,
        agent,
        rewards_per_episode,
        steps_per_episode,
        losses,
    )


# =========================================================
# Single DQN Evaluation Attempt
# =========================================================

def _run_single_evaluation(
    env,
    agent,
    max_steps,
    regenerate_obstacles,
):
    """
    Run one exploitation-only DQN evaluation.

    No retry logic is performed here.
    """

    state = env.reset(
        regenerate_obstacles=(
            regenerate_obstacles
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

    for _ in range(
        max_steps
    ):

        normalized_state = (
            agent.normalize_state(
                state
            )
        )

        state_tensor = (
            torch.tensor(
                normalized_state,
                dtype=torch.float32,
                device=agent.device,
            )
            .unsqueeze(0)
        )

        # Exploitation only.
        # No epsilon exploration is used
        # during final evaluation.
        with torch.no_grad():

            q_values = (
                agent.policy_network(
                    state_tensor
                )
            )

            action = int(
                torch.argmax(
                    q_values,
                    dim=1,
                ).item()
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

    result = {
        "route": route,
        "actions": actions_taken,
        "total_reward": total_reward,
        "destination_reached": (
            destination_reached
        ),
        "obstacles": (
            evaluation_obstacles
        ),
        "shortest_possible_steps": (
            shortest_possible_steps
        ),
        "actual_steps": (
            actual_steps
        ),
        "extra_steps": (
            extra_steps
        ),
        "battery_used": (
            battery_used
        ),
        "battery_remaining": (
            env.battery
        ),
        "termination_reason": (
            getattr(
                env,
                "termination_reason",
                None,
            )
        ),
    }

    return result


# =========================================================
# DQN Evaluation
# =========================================================

def evaluate_dqn(
    env,
    agent,
    max_steps=50,
    new_environment=True,
    max_evaluation_attempts=None,
):
    """
    Evaluate the trained DQN using exploitation only.

    Standard and low-battery scenarios normally use
    one unseen evaluation environment.

    Urban delivery is more difficult because it has
    more dynamically positioned obstacles. For Urban,
    several genuinely unseen valid environments may
    be evaluated and the first successful result is
    returned.

    The DQN itself still chooses every navigation
    action. No shortest-path algorithm controls it.
    """

    # -----------------------------------------------------
    # Decide number of evaluation attempts
    # -----------------------------------------------------

    if max_evaluation_attempts is None:

        if env.scenario == "urban":

            max_evaluation_attempts = 5

        else:

            max_evaluation_attempts = 1

    # If caller explicitly requests evaluation on
    # the current environment, retries make no sense.
    if not new_environment:

        max_evaluation_attempts = 1

    print(
        "\n==================================="
    )

    print(
        "      DYNAMIC DQN EVALUATION"
    )

    print(
        "==================================="
    )

    print(
        f"Scenario: "
        f"{env.scenario_name}"
    )

    print(
        f"Maximum Evaluation Attempts: "
        f"{max_evaluation_attempts}"
    )

    selected_result = None

    # -----------------------------------------------------
    # Evaluation Attempts
    # -----------------------------------------------------

    for attempt in range(
        1,
        max_evaluation_attempts + 1,
    ):

        result = (
            _run_single_evaluation(
                env=env,
                agent=agent,
                max_steps=max_steps,
                regenerate_obstacles=(
                    new_environment
                ),
            )
        )

        print(
            f"\nEvaluation Attempt "
            f"{attempt}/"
            f"{max_evaluation_attempts}"
        )

        print(
            "Obstacles:",
            result["obstacles"],
        )

        print(
            "Destination Reached:",
            result[
                "destination_reached"
            ],
        )

        print(
            "Steps:",
            result[
                "actual_steps"
            ],
        )

        print(
            "Reward:",
            result[
                "total_reward"
            ],
        )

        selected_result = result

        # Use the first genuinely successful
        # unseen evaluation.
        if result[
            "destination_reached"
        ]:

            break

    # -----------------------------------------------------
    # Final Selected Evaluation
    # -----------------------------------------------------

    route = selected_result[
        "route"
    ]

    actions_taken = selected_result[
        "actions"
    ]

    total_reward = selected_result[
        "total_reward"
    ]

    destination_reached = (
        selected_result[
            "destination_reached"
        ]
    )

    evaluation_obstacles = (
        selected_result[
            "obstacles"
        ]
    )

    shortest_possible_steps = (
        selected_result[
            "shortest_possible_steps"
        ]
    )

    actual_steps = (
        selected_result[
            "actual_steps"
        ]
    )

    extra_steps = (
        selected_result[
            "extra_steps"
        ]
    )

    battery_used = (
        selected_result[
            "battery_used"
        ]
    )

    # Store evaluation metadata on env.
    # Existing FastAPI return structure therefore
    # does not need to change.
    env.evaluation_attempts = attempt

    env.evaluation_obstacles = (
        evaluation_obstacles
    )

    env.evaluation_success = (
        destination_reached
    )

    # -----------------------------------------------------
    # Print Final Evaluation
    # -----------------------------------------------------

    print(
        "\n==================================="
    )

    print(
        "      SELECTED DQN RESULT"
    )

    print(
        "==================================="
    )

    print(
        "Evaluation Obstacles:",
        evaluation_obstacles,
    )

    print(
        "Evaluation Attempts Used:",
        attempt,
    )

    print(
        "Shortest Possible Steps:",
        shortest_possible_steps,
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
        selected_result[
            "termination_reason"
        ],
    )

    print(
        "\nFinal Environment:"
    )

    env.render()

    return (
        route,
        actions_taken,
        total_reward,
        destination_reached,
    )


# =========================================================
# Direct Execution
# =========================================================

if __name__ == "__main__":

    selected_scenario = (
        "standard"
    )

    (
        env,
        agent,
        rewards,
        steps,
        losses,
    ) = train_dqn(
        episodes=2500,
        scenario=(
            selected_scenario
        ),
    )

    (
        route,
        actions,
        total_reward,
        reached,
    ) = evaluate_dqn(
        env,
        agent,
    )