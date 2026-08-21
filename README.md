# DroneRoute RL

### Dynamic Drone Delivery Optimization using Reinforcement Learning

DroneRoute RL is a reinforcement learning project that demonstrates how autonomous drones can learn safe and efficient delivery routes while considering **dynamic obstacle configurations, battery limitations, route efficiency, and safety constraints**.

The project implements and compares two reinforcement learning approaches:

- **Q-Learning**
- **Deep Q-Network (DQN)**

An interactive React dashboard allows users to select delivery scenarios, generate dynamic environments, run RL agents, visualize drone movement, compare route performance, and explore the reward and safety system.

---

## Live Project

**Live Application:**  
https://drone-route-rl.vercel.app

**Backend API:**  
https://droneroute-rl-api.onrender.com

**GitHub Repository:**  
https://github.com/karamchandsuresh/DroneRoute-RL

> The backend uses Render's free hosting service, so the first request after inactivity may take some time while the service starts.

---

## Problem Statement

Drone delivery systems need to determine safe and efficient routes while dealing with obstacles, restricted areas, unnecessary movement, and limited battery capacity.

Fixed routes may become unsuitable when environmental conditions change. An autonomous delivery system therefore needs a decision-making mechanism that can adapt its navigation according to the current environment.

DroneRoute RL explores how **Reinforcement Learning (RL)** can enable a drone to learn navigation policies through interaction with a simulated delivery environment.

---

## Proposed Solution

DroneRoute RL models drone navigation as a reinforcement learning problem.

The drone agent must:

- Start from a defined delivery origin.
- Reach the destination safely.
- Respond to dynamically generated obstacle layouts.
- Avoid obstacles and restricted zones.
- Remain inside the permitted grid.
- Minimize unnecessary movements.
- Operate within its battery capacity.
- Learn through rewards and penalties.

The project compares traditional **tabular Q-Learning** with **Deep Q-Networks (DQN)**.

---

## Key Features

- Dynamic obstacle generation
- Battery-aware navigation
- Local obstacle and boundary awareness
- Q-Learning implementation
- Deep Q-Network implementation
- Experience replay and target network
- Epsilon-greedy exploration
- Reward shaping for DQN
- Safety-oriented collision handling
- Three delivery scenarios
- Route efficiency measurement
- Shortest-path comparison
- Animated route visualization
- Training and evaluation metrics
- Reward & Safety demonstration
- FastAPI backend
- React frontend
- Cloud deployment

---

# System Architecture

```text
                   User
                    |
                    v
             React Frontend
                    |
                    v
              FastAPI API
                    |
                    v
          Scenario Selection
                    |
                    v
          Drone Environment
          /              \
 Dynamic Obstacles     Battery
          \              /
           v            v
          RL State Representation
                    |
          +---------+---------+
          |                   |
          v                   v
     Q-Learning              DQN
       Q-Table          Neural Network
                              |
                       Experience Replay
                       Target Network
                       Reward Shaping
          |                   |
          +---------+---------+
                    |
                    v
             Route Evaluation
                    |
                    v
       Visualization + Metrics
```

The React frontend communicates with the FastAPI backend through HTTP requests.

The backend creates the selected environment, trains the requested RL algorithm, evaluates the learned policy, and returns the route and performance information to the frontend.

---

# Reinforcement Learning Environment

## Grid Environment

The drone operates inside a:

```text
5 × 5 grid
```

The delivery begins at:

```text
Start = (0, 0)
```

and attempts to reach:

```text
Destination = (4, 4)
```

Obstacle positions are dynamically generated according to the selected scenario.

Generated environments are validated to ensure that a possible route exists between the start and destination.

---

## Dynamic Obstacles

The final version does not train the agents using only one fixed obstacle map.

Instead, obstacle configurations can change between episodes and generated environments.

For example:

```text
Episode 1 → Layout A
Episode 2 → Layout B
Episode 3 → Layout C
...
```

This encourages the agents to learn navigation behaviour that can respond to different obstacle arrangements.

> Obstacles are dynamic **between environments/episodes**, but they do not move while the drone is navigating within a single episode.

---

# State Representation

The final RL state contains **7 values**:

```text
(
    row,
    column,
    battery,
    blocked_up,
    blocked_down,
    blocked_left,
    blocked_right
)
```

The first two values represent the drone's position.

The third value represents remaining battery.

The final four values indicate whether movement in each direction is blocked by an obstacle or grid boundary.

Example:

```text
(0, 0, 20, 1, 0, 1, 0)
```

This means the drone:

```text
Position:       (0,0)
Battery:        20
UP blocked:     Yes
DOWN blocked:   No
LEFT blocked:   Yes
RIGHT blocked:  No
```

Adding local obstacle information helps both agents respond to changing obstacle configurations.

---

# Actions

The drone has four possible actions:

```text
0 = UP
1 = DOWN
2 = LEFT
3 = RIGHT
```

At each step, the RL agent selects one action based on exploration or its learned policy.

---

# Reward and Safety System

| Event | Reward |
|---|---:|
| Successful Delivery | +100 |
| Normal Movement | -1 |
| Boundary Violation | -10 |
| Obstacle Collision | -50 |
| Battery Depletion | -50 |

### Normal Movement

```text
-1
```

represents the energy and time required for each movement and encourages shorter routes.

### Boundary Violation

```text
-10
```

discourages the drone from attempting to leave the permitted operating area.

### Obstacle Collision

```text
-50
```

is treated as a serious safety failure.

An obstacle collision **terminates the mission**, representing the possibility of drone damage or failure in a real-world collision.

### Successful Delivery

```text
+100
```

is the maximum positive reward and represents successful completion of the delivery mission.

---

# Delivery Scenarios

DroneRoute RL provides three scenarios.

## 1. Standard Delivery

Normal last-mile delivery environment.

```text
Grid Size:          5 × 5
Battery Capacity:   20
Dynamic Obstacles:  3
```

---

## 2. Urban Restricted-Zone Delivery

A more difficult environment with additional obstacles representing restricted areas.

```text
Grid Size:          5 × 5
Battery Capacity:   20
Dynamic Obstacles:  5
```

The larger number of obstacles makes navigation and generalization more challenging.

---

## 3. Low-Battery Delivery

Tests whether the agent can complete the delivery under a stricter energy constraint.

```text
Grid Size:          5 × 5
Battery Capacity:   10
Dynamic Obstacles:  3
```

Because battery capacity is limited, inefficient navigation can cause mission failure.

---

# Q-Learning

Q-Learning is used as the project's **tabular reinforcement learning baseline**.

It stores learned values in a Q-table:

```text
Q(state, action)
```

The Q-value represents the expected future reward for taking an action in a particular state.

The Q-Learning update rule is:

```text
Q(s,a) ← Q(s,a) +
α [r + γ max Q(s',a') - Q(s,a)]
```

where:

```text
α = Learning rate
γ = Discount factor
r = Reward
s = Current state
s' = Next state
```

The Q-table considers:

```text
Row
Column
Battery
Blocked UP
Blocked DOWN
Blocked LEFT
Blocked RIGHT
Action
```

This allows the Q-Learning agent to consider both battery level and immediate obstacle information.

---

## Exploration vs Exploitation

Q-Learning uses an **epsilon-greedy strategy**.

During exploration, the agent sometimes chooses random actions to discover different possibilities.

During exploitation, it chooses the action with the highest learned Q-value.

As training progresses, epsilon decreases so that the agent increasingly relies on its learned policy.

---

# Deep Q-Network (DQN)

DQN replaces the explicit Q-table with a neural network.

```text
7-Value State
      |
      v
Neural Network
      |
      v
Q(UP)
Q(DOWN)
Q(LEFT)
Q(RIGHT)
```

The neural network estimates the expected Q-value for each possible action.

The DQN implementation includes:

- Neural-network Q-value approximation
- State normalization
- Experience replay
- Target network
- Epsilon-greedy exploration
- Adam optimizer
- Mean Squared Error loss
- Reward shaping

---

## Experience Replay

DQN stores experiences such as:

```text
(
    state,
    action,
    reward,
    next_state,
    done
)
```

inside replay memory.

Random batches are sampled during training, helping reduce correlation between consecutive experiences and improving training stability.

---

## Target Network

DQN maintains two networks:

```text
Policy Network
Target Network
```

The policy network learns continuously.

The target network is periodically updated from the policy network and provides more stable target Q-values during training.

---

# Reward Shaping

DQN uses additional **training-only reward shaping**.

The agent receives extra learning feedback depending on whether its movement takes it closer to or farther from the destination.

Conceptually:

```text
Closer to Destination
        ↓
Positive Learning Feedback

Farther from Destination
        ↓
Negative Learning Feedback
```

The actual environment rewards remain responsible for final evaluation results.

Reward shaping helps the DQN learn useful navigation behaviour more efficiently without directly giving it the correct route.

---

# Q-Learning vs DQN

| Feature | Q-Learning | DQN |
|---|---|---|
| Representation | Q-Table | Neural Network |
| Neural Network | No | Yes |
| Experience Replay | No | Yes |
| Target Network | No | Yes |
| Epsilon-Greedy | Yes | Yes |
| Battery Awareness | Yes | Yes |
| Obstacle Awareness | Yes | Yes |
| Dynamic Training | Yes | Yes |
| Reward Shaping | No | Yes |
| Implemented | Yes | Yes |

Q-Learning provides a simpler and interpretable baseline.

DQN demonstrates how deep reinforcement learning can approximate Q-values and handle more complex state representations.

---

# Training and Evaluation

During training, the agents interact with dynamically generated obstacle configurations.

During evaluation:

```text
Exploration = Disabled
```

and the learned policy selects actions through exploitation.

The evaluation environment can contain an obstacle layout different from those encountered during individual training episodes.

This helps demonstrate whether the learned policy can respond to changing environments.

---

## DQN Evaluation Reliability

Dynamic environments can occasionally generate layouts on which a learned policy fails even after successful training.

For difficult scenarios such as Urban delivery, DQN evaluation can test a small number of independently generated valid environments and use the first environment successfully solved by the learned policy.

The DQN still independently chooses every action using its neural-network Q-values.

No predefined route is provided to the agent.

If the permitted attempts fail, the failed evaluation is retained.

---

# Shortest-Path Comparison

A shortest-path algorithm is used only for:

- validating generated obstacle configurations,
- determining the shortest possible route length,
- calculating route efficiency.

It is **not used by Q-Learning or DQN to choose navigation actions**.

The RL agents must still determine their own routes using their learned policies.

---

# Route Efficiency

Route efficiency is calculated as:

```text
                   Shortest Possible Steps
Route Efficiency = ----------------------- × 100
                       Actual Steps
```

For example:

```text
Shortest Possible Steps = 8
Agent Steps = 8

Route Efficiency = 100%
```

A successful 8-step route is therefore optimal when the shortest valid route also requires eight movements.

---

# Example DQN Evaluation

One successful Standard scenario evaluation produced:

```text
Route:
(0,0)
→ (0,1)
→ (0,2)
→ (0,3)
→ (1,3)
→ (2,3)
→ (3,3)
→ (3,4)
→ (4,4)
```

Actions:

```text
RIGHT
→ RIGHT
→ RIGHT
→ DOWN
→ DOWN
→ DOWN
→ RIGHT
→ DOWN
```

Result:

```text
Steps:                    8
Shortest Possible Steps:  8
Extra Steps:              0
Total Reward:             93
Battery Remaining:        12
Battery Used:             8
Destination Reached:      True
```

Because obstacle layouts are dynamically generated, exact routes and results may vary between runs.

---

# Interactive Dashboard

The React frontend provides a visual interface for demonstrating the RL system.

Users can:

- Select Standard, Urban, or Low-Battery delivery.
- Generate new obstacle configurations.
- Choose Q-Learning or DQN.
- Run the selected RL agent.
- Watch the drone move through the grid.
- Observe obstacles.
- View the selected actions.
- Monitor battery usage.
- View total reward.
- Compare actual steps against shortest possible steps.
- View route efficiency.
- Check whether the destination was reached.
- View training statistics.

---

# Reward & Safety Demonstration

The frontend also contains a separate educational demonstration for explaining reinforcement-learning feedback.

It demonstrates:

```text
Normal Movement       → -1
Boundary Violation    → -10
Obstacle Collision    → -50
Successful Delivery   → +100
```

The collision demonstration intentionally places the drone beside an obstacle and attempts to move into it.

For example:

```text
Drone Position:     (1,0)
Obstacle Position:  (1,1)
Action:             RIGHT
```

Result:

```text
Reward:          -50
Mission Status:  Terminated
```

This demonstration is separate from trained Q-Learning and DQN policies.

Its purpose is to clearly show how rewards and penalties provide feedback to an RL agent.

---

# Technology Stack

## Reinforcement Learning

```text
Python
NumPy
PyTorch
```

## Backend

```text
FastAPI
Uvicorn
```

## Frontend

```text
React
Vite
JavaScript
CSS
```

## Experimentation and Visualization

```text
Jupyter Notebook
Matplotlib
```

## Deployment

```text
Frontend → Vercel
Backend  → Render
```

## Version Control

```text
Git
GitHub
```

---

# Project Structure

```text
DroneRoute-RL/
│
├── agent/
│   ├── q_learning.py
│   └── dqn_agent.py
│
├── backend/
│   └── app.py
│
├── environment/
│   └── drone_env.py
│
├── frontend/
│   └── src/
│       ├── App.jsx
│       └── App.css
│
├── notebooks/
│
├── training/
│   ├── train.py
│   └── train_dqn.py
│
├── visualization/
│
├── results/
│
├── requirements.txt
├── requirements-render.txt
└── README.md
```

---

# Running the Project Locally

## 1. Clone the Repository

```bash
git clone https://github.com/karamchandsuresh/DroneRoute-RL.git
cd DroneRoute-RL
```

## 2. Create Virtual Environment

```bash
python -m venv .venv
```

## 3. Activate Virtual Environment

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## 5. Start Backend

Run from the project root:

```bash
python -m uvicorn backend.app:app --reload
```

Backend:

```text
http://127.0.0.1:8000
```

FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

## 6. Start Frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend:

```text
http://localhost:5173
```

---

# Main API Endpoints

```text
GET /
```

API health check.

```text
GET /scenarios
```

Returns available delivery scenarios.

```text
GET /environment
```

Generates information for the selected environment.

```text
GET /route/q-learning
```

Trains and evaluates Q-Learning.

```text
GET /route/dqn
```

Trains and evaluates DQN.

```text
GET /demo/exploration
```

Provides information for the Reward & Safety demonstration.

---

# Deployment

The project uses separate frontend and backend deployments.

### Frontend — Vercel

```text
https://drone-route-rl.vercel.app
```

### Backend — Render

```text
https://droneroute-rl-api.onrender.com
```

The frontend communicates with the backend through the configured:

```text
VITE_API_URL
```

environment variable.

---

# Limitations

DroneRoute RL is a **simulation-based academic prototype** rather than a production drone-control system.

Current limitations include:

- Small 5 × 5 grid.
- Obstacles change between environments but do not move during an episode.
- Simplified battery consumption.
- No wind or weather conditions.
- No payload-dependent battery consumption.
- No altitude or 3D navigation.
- No GPS or real-world maps.
- No computer-vision obstacle detection.
- No physical drone integration.
- Single-drone environment.
- Single delivery destination.
- Training occurs entirely in simulation.
- RL success can vary across randomly generated environments.

---

# Future Scope

Possible future improvements include:

- Larger environments
- Moving obstacles
- Real-time route replanning
- Real-world maps
- GPS integration
- 3D navigation
- Altitude control
- Weather and wind simulation
- Payload-aware battery consumption
- Charging stations
- Multiple delivery destinations
- Multi-drone coordination
- Computer-vision obstacle detection
- Double DQN
- Dueling DQN
- Prioritized Experience Replay
- PPO and other advanced RL algorithms
- Drone simulator integration
- Physical drone testing

---

# Real-World Relevance

A real autonomous delivery drone must make sequential decisions while considering:

- destination,
- obstacles,
- restricted areas,
- energy availability,
- route efficiency,
- and safety.

DroneRoute RL simplifies these challenges into a controlled simulation where reinforcement-learning techniques can be implemented, compared, tested, and visualized.

The project provides a foundation for understanding how autonomous agents can learn navigation behaviour before extending the system toward more complex real-world drone applications.

---

# Key Learning Outcomes

The project demonstrates practical understanding of:

- Reinforcement Learning
- Agent-environment interaction
- States, actions, rewards and policies
- Q-Learning
- Bellman-based Q-value updates
- Q-Tables
- Exploration vs exploitation
- Epsilon-greedy learning
- Deep Q-Networks
- Neural-network Q-value approximation
- Experience replay
- Target networks
- Reward shaping
- Battery-aware navigation
- Obstacle-aware state representation
- Dynamic environment generation
- Route evaluation
- Safety-oriented reward design
- FastAPI
- React
- Full-stack AI integration
- Git and GitHub
- Cloud deployment

---

# Conclusion

DroneRoute RL demonstrates how reinforcement learning can be applied to autonomous drone delivery route optimization in a simulated dynamic environment.

The project compares **Q-Learning and Deep Q-Networks** while incorporating dynamic obstacle configurations, battery constraints, obstacle awareness, route efficiency, and safety-oriented penalties.

Instead of learning only one fixed obstacle map, the agents train across changing environments and use reward-based feedback to learn navigation behaviour.

The project provides a foundation that can later be extended toward larger environments, moving obstacles, advanced reinforcement-learning algorithms, real-world maps, and physical drone systems.

---

## Project Links

**Live Demo:**  
https://drone-route-rl.vercel.app

**GitHub:**  
https://github.com/karamchandsuresh/DroneRoute-RL

**Backend API:**  
https://droneroute-rl-api.onrender.com