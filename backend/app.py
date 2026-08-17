from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from environment.drone_env import DroneEnvironment
from agent.q_learning import QLearningAgent
from training.train_dqn import train_dqn, evaluate_dqn


app = FastAPI(
    title="DroneRoute RL API",
    description=(
        "Backend API for Drone Delivery Optimization "
        "using Reinforcement Learning."
    ),
    version="1.0.0"
)


# Allow React/Vite frontend to access the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


ACTION_NAMES = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT"
}


# ---------------------------------------------------------
# Health Check
# ---------------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "DroneRoute RL API is running",
        "status": "success"
    }


# ---------------------------------------------------------
# Environment Information
# ---------------------------------------------------------

@app.get("/environment")
def get_environment():
    env = DroneEnvironment()

    return {
        "grid_size": env.grid_size,
        "start": list(env.start_position),
        "destination": list(env.destination),
        "obstacles": [
            list(position)
            for position in env.obstacles
        ],
        "max_battery": env.max_battery
    }


# ---------------------------------------------------------
# Q-Learning Training
# ---------------------------------------------------------

def train_q_learning(episodes=1500):
    """
    Train the battery-aware Q-Learning agent.
    """

    env = DroneEnvironment()

    agent = QLearningAgent(
        grid_size=env.grid_size,
        max_battery=env.max_battery
    )

    successful_episodes = 0

    for episode in range(episodes):

        state = env.reset()

        for step in range(100):

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

            if done:

                if env.drone_position == env.destination:
                    successful_episodes += 1

                break

        agent.decay_epsilon()

    success_rate = (
        successful_episodes / episodes
    ) * 100

    return env, agent, success_rate


# ---------------------------------------------------------
# Q-Learning Route Endpoint
# ---------------------------------------------------------

@app.get("/route/q-learning")
def get_q_learning_route():
    """
    Train Q-Learning and return the learned route.
    """

    env, agent, success_rate = train_q_learning()

    state = env.reset()

    route = [
        list(env.drone_position)
    ]

    actions = []

    total_reward = 0

    for step in range(50):

        row, col, battery = state

        action = int(
            agent.q_table[
                row,
                col,
                battery
            ].argmax()
        )

        next_state, reward, done = env.step(action)

        actions.append(
            ACTION_NAMES[action]
        )

        route.append(
            list(env.drone_position)
        )

        total_reward += reward

        state = next_state

        if done:
            break

    destination_reached = (
        env.drone_position
        == env.destination
    )

    return {
        "algorithm": "Q-Learning",
        "route": route,
        "actions": actions,
        "steps": len(actions),
        "total_reward": total_reward,
        "battery_remaining": env.battery,
        "battery_used": (
            env.max_battery - env.battery
        ),
        "destination_reached": destination_reached,
        "training_success_rate": round(
            success_rate,
            2
        )
    }


# ---------------------------------------------------------
# DQN Route Endpoint
# ---------------------------------------------------------

@app.get("/route/dqn")
def get_dqn_route():
    """
    Train the DQN agent and return its learned route.
    """

    (
        env,
        agent,
        rewards,
        steps,
        losses
    ) = train_dqn(
        episodes=1000
    )

    (
        route,
        actions,
        total_reward,
        destination_reached
    ) = evaluate_dqn(
        env,
        agent
    )

    return {
        "algorithm": "DQN",
        "route": [
            list(position)
            for position in route
        ],
        "actions": actions,
        "steps": len(actions),
        "total_reward": total_reward,
        "battery_remaining": env.battery,
        "battery_used": (
            env.max_battery - env.battery
        ),
        "destination_reached": destination_reached,
        "average_reward_last_100": round(
            sum(rewards[-100:]) / len(rewards[-100:]),
            2
        )
    }
    
@app.get("/demo/exploration")
def exploration_demo():
    """
    Demonstrate rewards and penalties using one
    continuous drone journey.

    The sequence includes:
    - boundary collision,
    - normal movement,
    - obstacle collision,
    - safe navigation,
    - destination reward.
    """

    env = DroneEnvironment()

    events = []

    cumulative_reward = 0
    step_number = 0

    action_names = {
        0: "UP",
        1: "DOWN",
        2: "LEFT",
        3: "RIGHT"
    }

    # Continuous demonstration sequence
    #
    # Start: (0, 0)
    #
    # UP       -> boundary collision
    # RIGHT    -> (0, 1)
    # DOWN     -> obstacle at (1, 1)
    # RIGHT    -> (0, 2)
    # RIGHT    -> (0, 3)
    # RIGHT    -> (0, 4)
    # DOWN     -> (1, 4)
    # DOWN     -> (2, 4)
    # DOWN     -> (3, 4)
    # DOWN     -> destination (4, 4)

    demonstration_actions = [
        0,  # UP - boundary
        3,  # RIGHT
        1,  # DOWN - obstacle
        3,  # RIGHT
        3,  # RIGHT
        3,  # RIGHT
        1,  # DOWN
        1,  # DOWN
        1,  # DOWN
        1   # DOWN - goal
    ]

    for action in demonstration_actions:

        step_number += 1

        state_before = env.drone_position

        row_change, col_change = env.actions[action]

        attempted_position = (
            state_before[0] + row_change,
            state_before[1] + col_change
        )

        next_state, reward, done = env.step(action)

        cumulative_reward += reward

        # Determine event type
        if done and env.drone_position == env.destination:
            event_type = "goal"

            message = (
                "Destination reached. Large positive "
                "reward received."
            )

        elif not env._is_inside_grid(attempted_position):
            event_type = "boundary"

            message = (
                "Boundary collision. Drone cannot "
                "leave the grid."
            )

        elif attempted_position in env.obstacles:
            event_type = "obstacle"

            message = (
                "Obstacle collision. Drone remains "
                "in the previous position."
            )

        else:
            event_type = "normal"

            message = (
                "Valid movement. Small step penalty received."
            )

        events.append({
            "type": event_type,
            "step": step_number,
            "action": action_names[action],
            "from": list(state_before),
            "attempted": list(attempted_position),
            "to": list(env.drone_position),
            "reward": reward,
            "cumulative_reward": cumulative_reward,
            "battery_used": (
                env.max_battery - env.battery
            ),
            "battery_remaining": env.battery,
            "destination_reached": (
                env.drone_position == env.destination
            ),
            "message": message
        })

        if done:
            break

    return {
        "demo": "RL Exploration and Reward Demonstration",
        "events": events
    }