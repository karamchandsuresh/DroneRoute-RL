# DroneRoute RL

### Drone Delivery Optimization using Reinforcement Learning

DroneRoute RL is a reinforcement learning project that demonstrates how intelligent agents can learn efficient drone delivery routes while considering obstacles, restricted zones, route efficiency, and limited battery capacity.

The system compares **Q-Learning** and **Deep Q-Network (DQN)** approaches across multiple delivery scenarios and provides an interactive web dashboard for visualizing the learned routes and reinforcement learning behaviour.

## Live Project

**Live Application:**  
https://drone-route-rl.vercel.app

**Backend API:**  
https://droneroute-rl-api.onrender.com

**GitHub Repository:**  
https://github.com/karamchandsuresh/DroneRoute-RL

> The backend is hosted on Render's free service, so the first request after a period of inactivity may take some time to start.

---

## Problem Statement

Drone delivery systems must determine efficient routes while dealing with practical constraints such as obstacles, restricted areas, unnecessary movement, and limited battery capacity.

Traditional fixed-route approaches may not adapt well when environmental conditions or delivery constraints change.

DroneRoute RL explores how **reinforcement learning** can allow a drone agent to learn navigation policies through interaction with a simulated environment and improve its delivery decisions based on rewards and penalties.

---

## Proposed Solution

DroneRoute RL models drone delivery as a reinforcement learning problem.

The drone interacts with a grid-based environment where it must:

- Start from a defined delivery origin.
- Navigate toward the customer destination.
- Avoid obstacles and restricted zones.
- Minimize unnecessary movements.
- Operate within the available battery capacity.
- Learn from rewards and penalties.
- Reach the destination using an efficient route.

Two reinforcement learning algorithms are implemented and compared:

1. **Q-Learning**
2. **Deep Q-Network (DQN)**

---

## Why This Project Is Useful

Drone delivery is fundamentally a decision-making problem.

A delivery drone must continuously decide which action should be taken based on its current position, environmental restrictions, and available energy.

DroneRoute RL demonstrates how reinforcement learning can support this decision-making process by learning policies that balance:

- Route efficiency
- Delivery success
- Obstacle avoidance
- Restricted-zone avoidance
- Battery consumption

The current project is a simulation rather than a real drone control system, but it provides a foundation for studying how RL-based navigation could be extended to more realistic autonomous delivery environments.

---

## System Architecture

```text
                     DroneRoute RL
                           |
              +------------+------------+
              |                         |
        React Frontend             FastAPI Backend
              |                         |
              |                 Scenario Selection
              |                         |
              |                  Drone Environment
              |                         |
              |              +----------+----------+
              |              |                     |
              |         Q-Learning                DQN
              |              |                     |
              |         Q-Table Learning      Neural Network
              |                                    |
              |                           Experience Replay
              |                           Target Network
              |                           Reward Shaping
              |                                    |
              +--------------- API ----------------+
                              |
                       Route Visualization
                              |
                    Metrics & RL Demonstration
```

---

## Reinforcement Learning Environment

The drone operates in a **5 × 5 grid environment**.

### State

The state is represented as:

```text
(row, column, battery)
```

This allows the agent to consider both its location and remaining energy when making decisions.

### Actions

The agent can perform four actions:

```text
0 = UP
1 = DOWN
2 = LEFT
3 = RIGHT
```

### Reward System

| Event | Reward / Penalty |
|---|---:|
| Destination reached | +100 |
| Normal movement | -1 |
| Boundary violation | -10 |
| Obstacle collision | -20 |
| Battery depletion | -50 |

The reward system encourages successful and efficient delivery while discouraging unsafe or unnecessary actions.

---

## Delivery Scenarios

DroneRoute RL provides three environments for evaluating the agents.

### 1. Standard Delivery

A normal last-mile delivery scenario with a moderate number of obstacles.

- Grid: 5 × 5
- Battery Capacity: 20
- Moderate obstacles

### 2. Urban Restricted-Zone Delivery

Represents a more constrained urban environment containing additional buildings or restricted areas.

- Grid: 5 × 5
- Battery Capacity: 20
- Increased number of obstacles/restricted zones

### 3. Low-Battery Delivery

Tests whether the agent can complete the delivery under a strict energy constraint.

- Grid: 5 × 5
- Battery Capacity: 10
- Energy-efficient navigation required

---

## Q-Learning

Q-Learning is used as the tabular reinforcement learning baseline.

The algorithm learns a **Q-value** for each state-action combination.

The agent uses an **epsilon-greedy strategy**:

- Exploration allows the drone to try different actions.
- Exploitation allows it to select actions with the highest learned Q-values.

Q-Learning works effectively for the project's relatively small discrete environment.

---

## Deep Q-Network (DQN)

DQN extends the reinforcement learning implementation by replacing the Q-table with a neural network.

The network receives the normalized state:

```text
[row, column, battery]
```

and predicts Q-values for:

```text
[UP, DOWN, LEFT, RIGHT]
```

The DQN implementation includes:

- Neural-network Q-value approximation
- Experience replay
- Target network
- Epsilon-greedy exploration
- State normalization
- Adam optimizer
- Mean Squared Error loss
- Reward shaping

DQN demonstrates how the project can move beyond tabular RL toward approaches capable of handling larger state representations.

---

## Reward Shaping Experiment

The Low-Battery scenario created an important reinforcement learning challenge.

Initially, DQN frequently exhausted its battery before discovering a successful route. Under the strict battery constraint, successful experiences were rare and the destination reward provided limited learning feedback.

Training-only **reward shaping** was therefore introduced.

The agent receives additional learning feedback when it moves closer to or farther from the destination.

The original environment reward system remains unchanged during final evaluation.

This helped DQN learn a successful route while maintaining the original delivery objective.

---

## Final Low-Battery DQN Result

The final DQN evaluation successfully completed the Low-Battery scenario.

| Metric | Result |
|---|---:|
| Battery Capacity | 10 |
| Route Length | 8 steps |
| Total Reward | 93 |
| Battery Used | 8 |
| Battery Remaining | 2 |
| Destination Reached | Yes |
| Last 100 Avg. Reward | 88.89 |
| Last 100 Avg. Steps | 8.11 |

### Learned Route

```text
(0,0)
   ↓
(0,1)
   ↓
(0,2)
   ↓
(0,3)
   ↓
(0,4)
   ↓
(1,4)
   ↓
(2,4)
   ↓
(3,4)
   ↓
(4,4)
```

Actions:

```text
RIGHT → RIGHT → RIGHT → RIGHT
→ DOWN → DOWN → DOWN → DOWN
```

The Manhattan distance between `(0,0)` and `(4,4)` is eight moves, so the final policy achieved a shortest 8-step route.

---

## Q-Learning vs DQN

| Feature | Q-Learning | DQN |
|---|---|---|
| Value representation | Q-Table | Neural Network |
| Suitable state space | Small/discrete | Larger/complex |
| Experience Replay | No | Yes |
| Target Network | No | Yes |
| Neural Network | No | Yes |
| Exploration | Epsilon-Greedy | Epsilon-Greedy |
| Implemented in DroneRoute RL | Yes | Yes |

Using both algorithms provides a useful comparison between traditional tabular reinforcement learning and deep reinforcement learning.

---

## Interactive Dashboard

The React dashboard provides:

- Scenario selection
- Q-Learning execution
- DQN execution
- Animated drone movement
- Learned route visualization
- Obstacles and restricted-zone visualization
- Battery usage
- Route steps
- Total reward
- Delivery success status
- Training statistics
- Reward and penalty exploration demonstration

The exploration demonstration visually explains how the drone receives feedback for normal movement, obstacle collisions, boundary violations, and successful delivery.

---

## Technology Stack

### Reinforcement Learning / Backend

- Python
- NumPy
- PyTorch
- FastAPI
- Uvicorn

### Visualization and Experimentation

- Matplotlib
- Jupyter Notebook

### Frontend

- React
- Vite
- JavaScript
- CSS

### Deployment

- Vercel — Frontend
- Render — FastAPI backend

### Version Control

- Git
- GitHub

---

## Project Structure

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
│   ├── src/
│   │   ├── App.jsx
│   │   └── App.css
│   └── ...
│
├── notebooks/
│   └── rl_experiments.ipynb
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

## Running the Project Locally

### 1. Clone the Repository

```bash
git clone https://github.com/karamchandsuresh/DroneRoute-RL.git
cd DroneRoute-RL
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Start the Backend

From the project root:

```bash
python -m uvicorn backend.app:app --reload
```

Backend:

```text
http://127.0.0.1:8000
```

### 6. Start the Frontend

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

## Deployment

The application is deployed using separate frontend and backend services.

### Frontend

Deployed on Vercel:

https://drone-route-rl.vercel.app

### Backend

FastAPI deployed on Render:

https://droneroute-rl-api.onrender.com

The frontend communicates with the deployed backend through the `VITE_API_URL` environment variable.

---

## Limitations

The current version is a controlled simulation and has several limitations:

- Uses a small 5 × 5 grid.
- Obstacles are static.
- Weather and wind are not simulated.
- Battery consumption is simplified.
- No GPS or real-world map integration.
- No physical drone hardware integration.
- Single-drone and single-delivery environment.
- Training occurs in simulation.

These limitations define opportunities for future development rather than representing a production-ready autonomous drone system.

---

## Future Scope

DroneRoute RL can be extended with:

- Larger and dynamic environments
- Moving obstacles
- Real-world map integration
- GPS-based navigation
- Weather and wind conditions
- Variable battery consumption
- Charging stations
- Emergency landing decisions
- Multiple delivery destinations
- Multi-drone coordination
- Real-time route replanning
- Advanced Deep RL algorithms
- Integration with drone simulators
- Physical drone testing

---

## Key Learning Outcomes

This project demonstrates practical implementation of:

- Reinforcement Learning
- Markov Decision Process concepts
- States, actions, rewards and policies
- Q-Learning
- Q-Tables
- Bellman-based value updates
- Exploration vs exploitation
- Epsilon-greedy learning
- Deep Q-Networks
- Experience replay
- Target networks
- Reward shaping
- Battery-aware state representation
- RL training and evaluation
- Full-stack integration
- API development
- Cloud deployment

---

## Conclusion

DroneRoute RL demonstrates how reinforcement learning can be applied to autonomous drone delivery route decision-making.

By comparing Q-Learning and DQN across standard, urban restricted-zone, and low-battery scenarios, the project demonstrates how an agent can learn efficient navigation policies through environmental interaction and reward-based feedback.

The project serves as a simulation-based foundation that can be extended toward more realistic autonomous delivery and intelligent route-planning systems.

---

## Links

**Live Demo:**  
https://drone-route-rl.vercel.app

**GitHub:**  
https://github.com/karamchandsuresh/DroneRoute-RL

**Backend API:**  
https://droneroute-rl-api.onrender.com