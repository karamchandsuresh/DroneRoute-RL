import { useEffect, useState } from "react";
import "./App.css";

const API_URL =
  import.meta.env.VITE_API_URL ||
  "http://127.0.0.1:8000";

const SCENARIOS = {
  standard: {
    label: "Standard Delivery",
    shortDescription:
      "Dynamic delivery environment with 3 randomly positioned obstacles.",
  },

  urban: {
    label: "Urban Restricted-Zone Delivery",
    shortDescription:
      "More constrained urban delivery with 5 randomly positioned restricted zones.",
  },

  low_battery: {
    label: "Low-Battery Delivery",
    shortDescription:
      "Energy-constrained delivery with dynamic obstacles and a 10-unit battery.",
  },
};

function App() {
  const [environment, setEnvironment] =
    useState(null);

  const [routeData, setRouteData] =
    useState(null);

  const [dronePosition, setDronePosition] =
    useState([0, 0]);

  const [algorithm, setAlgorithm] =
    useState("dqn");

  const [scenario, setScenario] =
    useState("standard");

  const [loading, setLoading] =
    useState(false);

  const [isAnimating, setIsAnimating] =
    useState(false);

  const [error, setError] =
    useState("");

  const [showRewardSystem, setShowRewardSystem] =
    useState(false);

  const [collisionDemo, setCollisionDemo] =
    useState(false);

  const [collisionResult, setCollisionResult] =
    useState(false);

  // ==================================================
  // LOAD ENVIRONMENT
  // ==================================================

  useEffect(() => {
    fetchEnvironment(scenario);
  }, [scenario]);

  const fetchEnvironment = async (
    selectedScenario
  ) => {
    try {
      setError("");

      const safeScenario =
        selectedScenario &&
        Object.prototype.hasOwnProperty.call(
          SCENARIOS,
          selectedScenario
        )
          ? selectedScenario
          : "standard";

      const response = await fetch(
        `${API_URL}/environment?scenario=${safeScenario}`
      );

      if (!response.ok) {
        throw new Error(
          "Failed to load environment."
        );
      }

      const data =
        await response.json();

      setEnvironment(data);

      setDronePosition(
        data.start
      );

      setRouteData(null);

      setCollisionDemo(false);
      setCollisionResult(false);
    } catch (err) {
      console.error(err);

      setError(
        "Could not connect to the DroneRoute RL backend."
      );
    }
  };

  // ==================================================
  // HELPERS
  // ==================================================

  const sleep = (milliseconds) => {
    return new Promise((resolve) =>
      setTimeout(
        resolve,
        milliseconds
      )
    );
  };

  const getAlgorithmLabel = () => {
    return algorithm === "q-learning"
      ? "Q-Learning"
      : "Deep Q-Network (DQN)";
  };

  // ==================================================
  // APPLY ROUTE ENVIRONMENT
  // ==================================================

  const applyRouteEnvironment = (
    data
  ) => {
    setEnvironment({
      scenario:
        data.scenario,

      name:
        data.scenario_name,

      description:
        data.scenario_description,

      grid_size:
        data.grid_size,

      start:
        data.start,

      destination:
        data.destination,

      obstacles:
        data.obstacles,

      obstacle_count:
        data.obstacles.length,

      max_battery:
        data.battery_capacity,

      dynamic_obstacles:
        true,

      shortest_possible_steps:
        data.shortest_possible_steps,
    });
  };

  // ==================================================
  // RUN RL AGENT
  // ==================================================

  const runAgent = async () => {
    if (
      !environment ||
      loading ||
      isAnimating
    ) {
      return;
    }

    try {
      setLoading(true);

      setError("");

      setRouteData(null);

      setCollisionDemo(false);
      setCollisionResult(false);

      setDronePosition(
        environment.start
      );

      const endpoint =
        algorithm === "q-learning"
          ? "/route/q-learning"
          : "/route/dqn";

      const response = await fetch(
        `${API_URL}${endpoint}?scenario=${scenario}`
      );

      if (!response.ok) {
        throw new Error(
          `${getAlgorithmLabel()} request failed.`
        );
      }

      const data =
        await response.json();

      applyRouteEnvironment(
        data
      );

      setRouteData(
        data
      );

      setDronePosition(
        data.route[0]
      );

      setLoading(false);

      setIsAnimating(true);

      for (
        const position
        of data.route
      ) {
        setDronePosition(
          position
        );

        await sleep(420);
      }

      setIsAnimating(false);
    } catch (err) {
      console.error(err);

      setLoading(false);

      setIsAnimating(false);

      setError(
        `Unable to generate the ${getAlgorithmLabel()} route.`
      );
    }
  };

  // ==================================================
  // GENERATE NEW ENVIRONMENT
  // ==================================================

  const generateNewEnvironment =
    async () => {
      if (
        loading ||
        isAnimating
      ) {
        return;
      }

      setCollisionDemo(false);
      setCollisionResult(false);

      await fetchEnvironment(
        scenario
      );
    };

  // ==================================================
  // CONTROLLED COLLISION DEMONSTRATION
  // ==================================================

  const demonstrateCollision =
    async () => {
      if (
        loading ||
        isAnimating
      ) {
        return;
      }

      /*
       * This is an isolated educational
       * visualization of the collision rule.
       *
       * It does NOT train Q-Learning or DQN
       * and does NOT modify their policies.
       *
       * Drone:    (1, 0)
       * Obstacle: (1, 1)
       *
       * The drone attempts RIGHT.
       */

      setRouteData(null);

      setCollisionDemo(true);
      setCollisionResult(false);

      setEnvironment({
        scenario:
          "collision_demo",

        name:
          "Obstacle Collision Demonstration",

        description:
          "Controlled visualization of the obstacle collision penalty.",

        grid_size:
          5,

        start:
          [1, 0],

        destination:
          [4, 4],

        obstacles:
          [[1, 1]],

        obstacle_count:
          1,

        max_battery:
          20,

        dynamic_obstacles:
          false,

        shortest_possible_steps:
          null,
      });

      setDronePosition(
        [1, 0]
      );

      /*
       * Pause so the audience can first
       * see the drone beside the obstacle.
       */

      await sleep(1200);

      /*
       * Collision result:
       *
       * The drone attempts to move RIGHT
       * into obstacle cell (1,1).
       *
       * It remains at (1,0), receives -50,
       * and the episode terminates.
       */

      setCollisionResult(
        true
      );
    };

  // ==================================================
  // RETURN FROM COLLISION DEMO
  // ==================================================

  const returnFromCollisionDemo =
    async () => {
      setCollisionDemo(false);

      setCollisionResult(false);

      await fetchEnvironment(
        scenario
      );
    };

  // ==================================================
  // SCENARIO CHANGE
  // ==================================================

  const handleScenarioChange = (
    newScenario
  ) => {
    if (
      loading ||
      isAnimating
    ) {
      return;
    }

    if (
      !newScenario ||
      !Object.prototype.hasOwnProperty.call(
        SCENARIOS,
        newScenario
      )
    ) {
      return;
    }

    setCollisionDemo(false);
    setCollisionResult(false);

    setShowRewardSystem(false);

    setScenario(
      newScenario
    );

    setRouteData(null);

    setError("");
  };

  // ==================================================
  // ALGORITHM CHANGE
  // ==================================================

  const handleAlgorithmChange = (
    event
  ) => {
    if (
      loading ||
      isAnimating
    ) {
      return;
    }

    setAlgorithm(
      event.target.value
    );

    setRouteData(null);

    setCollisionDemo(false);
    setCollisionResult(false);

    if (environment) {
      setDronePosition(
        environment.start
      );
    }
  };

  // ==================================================
  // GRID HELPERS
  // ==================================================

  const isSamePosition = (
    first,
    second
  ) => {
    return (
      first &&
      second &&
      first[0] === second[0] &&
      first[1] === second[1]
    );
  };

  const isObstacle = (
    row,
    col
  ) => {
    if (
      !environment ||
      !environment.obstacles
    ) {
      return false;
    }

    return environment.obstacles.some(
      ([
        obstacleRow,
        obstacleCol,
      ]) =>
        obstacleRow === row &&
        obstacleCol === col
    );
  };

  const isRouteCell = (
    row,
    col
  ) => {
    if (
      !routeData ||
      collisionDemo
    ) {
      return false;
    }

    return routeData.route.some(
      ([
        routeRow,
        routeCol,
      ]) =>
        routeRow === row &&
        routeCol === col
    );
  };

  const getCellClass = (
    row,
    col
  ) => {
    const classes = [
      "grid-cell",
    ];

    if (
      isObstacle(
        row,
        col
      )
    ) {
      classes.push(
        "obstacle"
      );
    }

    if (
      isSamePosition(
        [row, col],
        environment.destination
      )
    ) {
      classes.push(
        "destination"
      );
    }

    if (
      isSamePosition(
        [row, col],
        environment.start
      )
    ) {
      classes.push(
        "start"
      );
    }

    if (
      isRouteCell(
        row,
        col
      )
    ) {
      classes.push(
        "route"
      );
    }

    if (
      isSamePosition(
        [row, col],
        dronePosition
      )
    ) {
      classes.push(
        "drone"
      );
    }

    return classes.join(
      " "
    );
  };

  const getCellContent = (
    row,
    col
  ) => {
    if (
      isSamePosition(
        [row, col],
        dronePosition
      )
    ) {
      return "🚁";
    }

    if (
      isObstacle(
        row,
        col
      )
    ) {
      return "✕";
    }

    if (
      isSamePosition(
        [row, col],
        environment.destination
      )
    ) {
      return "G";
    }

    if (
      isSamePosition(
        [row, col],
        environment.start
      )
    ) {
      return "S";
    }

    return "";
  };

  // ==================================================
  // METRICS
  // ==================================================

  const getStepsValue = () => {
    if (routeData) {
      return routeData.steps;
    }

    return "--";
  };

  const getRewardValue = () => {
    if (routeData) {
      return routeData.total_reward;
    }

    return "--";
  };

  const getBatteryUsedValue = () => {
    if (routeData) {
      return (
        `${routeData.battery_used}/` +
        `${routeData.battery_capacity}`
      );
    }

    return "--";
  };

  const getSuccessValue = () => {
    if (routeData) {
      return routeData.destination_reached
        ? "Yes"
        : "No";
    }

    return "--";
  };

  // ==================================================
  // LOADING SCREEN
  // ==================================================

  if (!environment) {
    return (
      <div className="loading-screen">

        <h2>
          DroneRoute RL
        </h2>

        <p>
          {error ||
            "Loading dynamic environment..."}
        </p>

      </div>
    );
  }

  // ==================================================
  // USER INTERFACE
  // ==================================================

  return (
    <div className="app">

      {/* ===============================================
          HEADER
      =============================================== */}

      <header className="header">

        <div>

          <p className="eyebrow">
            REINFORCEMENT LEARNING PROJECT
          </p>

          <h1>
            DroneRoute RL
          </h1>

          <p className="subtitle">
            Dynamic Drone Delivery Optimization
            using Reinforcement Learning
          </p>

        </div>

        <div className="status-badge">
          API Connected
        </div>

      </header>

      {/* ===============================================
          SCENARIO SECTION
      =============================================== */}

      <section className="scenario-section">

        <div className="scenario-heading">

          <div>

            <p className="section-label">
              DYNAMIC DELIVERY ENVIRONMENT
            </p>

            <h2>
              Choose Delivery Scenario
            </h2>

          </div>

          <span className="scenario-badge">
            Dynamic Obstacles
          </span>

        </div>

        <p className="scenario-introduction">
          Obstacle positions change between
          generated environments, requiring
          the RL agent to adapt its route
          instead of memorizing one fixed map.
        </p>

        <div className="scenario-options">

          {Object.entries(
            SCENARIOS
          ).map(
            ([
              scenarioKey,
              scenarioData,
            ]) => (

              <button
                type="button"
                key={scenarioKey}
                className={`scenario-card ${
                  scenario === scenarioKey
                    ? "active-scenario"
                    : ""
                }`}
                onClick={() =>
                  handleScenarioChange(
                    scenarioKey
                  )
                }
                disabled={
                  loading ||
                  isAnimating ||
                  collisionDemo
                }
              >

                <strong>
                  {scenarioData.label}
                </strong>

                <span>
                  {
                    scenarioData.shortDescription
                  }
                </span>

              </button>

            )
          )}

        </div>

        <div className="scenario-details">

          <div>

            <span>
              Scenario
            </span>

            <strong>
              {environment.name ||
                SCENARIOS[
                  scenario
                ].label}
            </strong>

          </div>

          <div>

            <span>
              Battery
            </span>

            <strong>
              {environment.max_battery} units
            </strong>

          </div>

          <div>

            <span>
              Obstacles
            </span>

            <strong>
              {
                environment.obstacles.length
              }
            </strong>

          </div>

          <div>

            <span>
              Shortest Path
            </span>

            <strong>
              {
                environment.shortest_possible_steps ??
                "--"
              }{" "}
              steps
            </strong>

          </div>

        </div>

        {!collisionDemo && (

          <button
            type="button"
            className="demo-button"
            onClick={
              generateNewEnvironment
            }
            disabled={
              loading ||
              isAnimating
            }
          >
            Generate New Obstacle Layout
          </button>

        )}

      </section>

      {/* ===============================================
          MAIN DASHBOARD
      =============================================== */}

      <main className="dashboard">

        {/* =============================================
            ENVIRONMENT GRID
        ============================================= */}

        <section className="panel environment-panel">

          <div className="panel-heading">

            <div>

              <p className="section-label">
                {collisionDemo
                  ? "SAFETY DEMONSTRATION"
                  : "ENVIRONMENT"}
              </p>

              <h2>
                {collisionDemo
                  ? "Obstacle Collision"
                  : "Delivery Grid"}
              </h2>

            </div>

            <span>
              {environment.grid_size} ×{" "}
              {environment.grid_size}
            </span>

          </div>

          <div
            className="grid"
            style={{
              gridTemplateColumns:
                `repeat(${environment.grid_size}, 1fr)`,
            }}
          >

            {Array.from(
              {
                length:
                  environment.grid_size,
              },
              (_, row) =>
                Array.from(
                  {
                    length:
                      environment.grid_size,
                  },
                  (_, col) => (

                    <div
                      className={
                        getCellClass(
                          row,
                          col
                        )
                      }
                      key={`${row}-${col}`}
                    >

                      {
                        getCellContent(
                          row,
                          col
                        )
                      }

                    </div>

                  )
                )
            )}

          </div>

          <div className="legend">

            <span>
              🚁 Drone
            </span>

            <span>
              S Start
            </span>

            <span>
              G Goal
            </span>

            <span>
              ✕ Obstacle
            </span>

          </div>

          {collisionDemo ? (

            <div className="description">

              {!collisionResult ? (

                <p>
                  The drone is positioned at
                  (1,0). An obstacle is directly
                  beside it at (1,1). The drone
                  attempts to move RIGHT.
                </p>

              ) : (

                <>
                  <p>
                    <strong>
                      Collision detected.
                    </strong>
                  </p>

                  <p>
                    The drone attempted to enter
                    obstacle cell (1,1), received
                    a -50 penalty and remained at
                    (1,0).
                  </p>

                  <p>
                    <strong>
                      Mission terminated.
                    </strong>
                  </p>
                </>

              )}

            </div>

          ) : (

            <p className="description">
              The displayed obstacles belong to
              the current environment. Running an
              RL agent evaluates it on a newly
              generated valid obstacle layout.
            </p>

          )}

        </section>

        {/* =============================================
            RL CONTROL PANEL
        ============================================= */}

        <section className="panel control-panel">

          <p className="section-label">
            RL AGENT
          </p>

          <h2>
            {getAlgorithmLabel()}
          </h2>

          <p className="description">
            Train the selected reinforcement
            learning agent across changing
            obstacle layouts and evaluate its
            policy on a new environment.
          </p>

          <div className="algorithm-control">

            <label htmlFor="algorithm">
              Reinforcement Learning Algorithm
            </label>

            <select
              id="algorithm"
              value={algorithm}
              onChange={
                handleAlgorithmChange
              }
              disabled={
                loading ||
                isAnimating ||
                collisionDemo
              }
            >

              <option value="q-learning">
                Q-Learning
              </option>

              <option value="dqn">
                Deep Q-Network (DQN)
              </option>

            </select>

          </div>

          <button
            className="run-button"
            onClick={
              runAgent
            }
            disabled={
              loading ||
              isAnimating ||
              collisionDemo
            }
          >

            {loading
              ? `Training ${getAlgorithmLabel()}...`
              : isAnimating
              ? "Drone Moving..."
              : `Run ${getAlgorithmLabel()}`}

          </button>

          {algorithm === "dqn" && (

            <p className="training-note">
              DQN uses a neural network,
              experience replay, a target
              network and reward shaping.
              Training may take longer on CPU.
            </p>

          )}

          {error && (

            <p className="error-message">
              {error}
            </p>

          )}

          {/* ===========================================
              PRIMARY RL METRICS
          =========================================== */}

          <div className="metrics">

            <Metric
              title="Steps"
              value={
                getStepsValue()
              }
            />

            <Metric
              title="Reward"
              value={
                getRewardValue()
              }
            />

            <Metric
              title="Battery Used"
              value={
                getBatteryUsedValue()
              }
            />

            <Metric
              title="Success"
              value={
                getSuccessValue()
              }
            />

          </div>

          {/* ===========================================
              ROUTE DETAILS
          =========================================== */}

          {routeData && (

            <div className="route-details">

              <h3>
                Evaluation Environment
              </h3>

              <p>
                Dynamic obstacle configuration
                generated after training.
              </p>

              <h3>
                Algorithm
              </h3>

              <p>
                {routeData.algorithm}
              </p>

              <h3>
                Learned Actions
              </h3>

              <p>
                {
                  routeData.actions.join(
                    " → "
                  )
                }
              </p>

              <h3>
                Shortest Possible Steps
              </h3>

              <p>
                {
                  routeData.shortest_possible_steps
                }
              </p>

              <h3>
                Agent Steps
              </h3>

              <p>
                {
                  routeData.steps
                }
              </p>

              <h3>
                Extra Steps
              </h3>

              <p>
                {
                  routeData.extra_steps ??
                  "Not available"
                }
              </p>

              <h3>
                Route Efficiency
              </h3>

              <p>
                {
                  routeData.route_efficiency !==
                  null &&
                  routeData.route_efficiency !==
                  undefined
                    ? `${routeData.route_efficiency}%`
                    : "Not available"
                }
              </p>

              <h3>
                Training Success Rate
              </h3>

              <p>
                {
                  routeData.training_success_rate
                }
                %
              </p>

              <h3>
                Last 100 Average Reward
              </h3>

              <p>
                {
                  routeData.average_reward_last_100
                }
              </p>

              <h3>
                Last 100 Average Steps
              </h3>

              <p>
                {
                  routeData.average_steps_last_100
                }
              </p>

            </div>

          )}

          {/* ===========================================
              REWARD AND SAFETY SYSTEM
          =========================================== */}

          <div className="demo-section">

            <div className="demo-divider" />

            <p className="section-label">
              LEARNING DEMONSTRATION
            </p>

            <h3>
              Reward & Safety System
            </h3>

            <p className="demo-description">
              Reinforcement learning improves
              through rewards and penalties.
              Unsafe actions receive stronger
              penalties so the agent learns to
              prefer safer routes.
            </p>

            <button
              type="button"
              className="demo-button"
              onClick={() =>
                setShowRewardSystem(
                  !showRewardSystem
                )
              }
              disabled={
                loading ||
                isAnimating ||
                collisionDemo
              }
            >

              {showRewardSystem
                ? "Hide Reward System"
                : "Show Reward System"}

            </button>

            {showRewardSystem && (

              <div className="reward-system">

                <RewardCard
                  title="Normal Movement"
                  reward="-1"
                  description="A small cost represents the time and battery energy used during each navigation step."
                />

                <RewardCard
                  title="Boundary Violation"
                  reward="-10"
                  description="The drone receives a larger penalty if it attempts to leave the permitted delivery area."
                />

                <RewardCard
                  title="Obstacle Collision"
                  reward="-50"
                  description="A collision is treated as a serious safety failure. The episode immediately terminates."
                  danger
                />

                <RewardCard
                  title="Successful Delivery"
                  reward="+100"
                  description="Reaching the destination successfully provides the largest positive reward."
                  positive
                />

                {/* =====================================
                    COLLISION EXPLORATION
                ===================================== */}

                <div className="demo-event-card">

                  <div className="demo-event-header">

                    <strong>
                      Interactive Collision Exploration
                    </strong>

                    <span className="reward-badge negative-reward">
                      -50
                    </span>

                  </div>

                  <p className="event-message">
                    Demonstrate what happens when
                    the drone attempts to enter a
                    cell occupied by an obstacle.
                  </p>

                  {!collisionDemo && (

                    <button
                      type="button"
                      className="demo-button"
                      onClick={
                        demonstrateCollision
                      }
                      disabled={
                        loading ||
                        isAnimating
                      }
                    >
                      Demonstrate Obstacle Collision
                    </button>

                  )}

                  {collisionDemo &&
                    !collisionResult && (

                    <div className="event-message">

                      <p>
                        <strong>
                          Drone Position:
                        </strong>{" "}
                        (1,0)
                      </p>

                      <p>
                        <strong>
                          Obstacle Position:
                        </strong>{" "}
                        (1,1)
                      </p>

                      <p>
                        <strong>
                          Action:
                        </strong>{" "}
                        RIGHT
                      </p>

                      <p>
                        Attempting movement...
                      </p>

                    </div>

                  )}

                  {collisionDemo &&
                    collisionResult && (

                    <div className="event-message">

                      <p>
                        <strong>
                          Action:
                        </strong>{" "}
                        RIGHT
                      </p>

                      <p>
                        <strong>
                          From:
                        </strong>{" "}
                        (1,0)
                      </p>

                      <p>
                        <strong>
                          Attempted Position:
                        </strong>{" "}
                        (1,1)
                      </p>

                      <p>
                        <strong>
                          Obstacle:
                        </strong>{" "}
                        Detected
                      </p>

                      <p>
                        <strong>
                          Drone Position:
                        </strong>{" "}
                        (1,0)
                      </p>

                      <p>
                        <strong>
                          Reward:
                        </strong>{" "}
                        -50
                      </p>

                      <p>
                        <strong>
                          Mission Status:
                        </strong>{" "}
                        Terminated
                      </p>

                      <p>
                        The drone does not continue
                        toward the destination after
                        the collision.
                      </p>

                      <button
                        type="button"
                        className="demo-button"
                        onClick={
                          returnFromCollisionDemo
                        }
                      >
                        Return to Environment
                      </button>

                    </div>

                  )}

                </div>

                <div className="safety-rule">

                  <strong>
                    Safety Rule
                  </strong>

                  <p>
                    Obstacle collision immediately
                    terminates the episode because
                    a physical collision could
                    damage or crash a real drone.
                  </p>

                </div>

                <p className="demo-description">
                  This section demonstrates the
                  reward structure and collision
                  safety rule. Actual learned
                  navigation is demonstrated
                  separately by running Q-Learning
                  or DQN.
                </p>

              </div>

            )}

          </div>

        </section>

      </main>

      {/* ===============================================
          PROJECT SUMMARY
      =============================================== */}

      <section className="real-world-section">

        <div className="section-intro">

          <p className="section-label">
            PROJECT SIGNIFICANCE
          </p>

          <h2>
            Why Dynamic DroneRoute RL Matters
          </h2>

          <p>
            Instead of learning one predefined
            obstacle map, the system trains
            across changing valid environments.
            The agent receives local obstacle
            information together with position
            and battery state, allowing it to
            adapt navigation decisions to the
            current delivery conditions.
          </p>

        </div>

        <div className="benefits-grid">

          <BenefitCard
            icon="↗"
            title="Dynamic Routing"
            description="Obstacle positions change between environments, so the agent must adapt instead of memorizing one fixed route."
          />

          <BenefitCard
            icon="◎"
            title="Obstacle Awareness"
            description="The RL state includes blocked directions, allowing decisions to respond to surrounding obstacles and boundaries."
          />

          <BenefitCard
            icon="⚡"
            title="Battery Awareness"
            description="Every attempted movement consumes energy, encouraging efficient navigation under battery constraints."
          />

          <BenefitCard
            icon="◈"
            title="Q-Learning vs DQN"
            description="The project compares tabular reinforcement learning with neural-network-based Deep Q-Learning in dynamic environments."
          />

        </div>

        <div className="project-conclusion">

          <p className="section-label">
            CURRENT SCOPE
          </p>

          <h2>
            Simulation-Based Prototype
          </h2>

          <p>
            DroneRoute RL currently uses a
            simplified 5 × 5 simulation with
            randomly generated obstacle
            configurations between episodes.
            Obstacles remain fixed during one
            episode. Future work can extend the
            system to moving obstacles, real
            maps, GPS, weather, payload effects
            and physical drone systems.
          </p>

        </div>

      </section>

    </div>
  );
}

// ==================================================
// REUSABLE COMPONENTS
// ==================================================

function Metric({
  title,
  value,
}) {
  return (
    <div className="metric-card">

      <span>
        {title}
      </span>

      <strong>
        {value}
      </strong>

    </div>
  );
}

function RewardCard({
  title,
  reward,
  description,
  danger = false,
  positive = false,
}) {
  return (
    <div
      className={`demo-event-card ${
        danger
          ? "reward-danger"
          : positive
          ? "reward-positive"
          : ""
      }`}
    >

      <div className="demo-event-header">

        <strong>
          {title}
        </strong>

        <span
          className={`reward-badge ${
            positive
              ? "positive-reward"
              : reward === "-1"
              ? "step-reward"
              : "negative-reward"
          }`}
        >
          Reward {reward}
        </span>

      </div>

      <p className="event-message">
        {description}
      </p>

    </div>
  );
}

function BenefitCard({
  icon,
  title,
  description,
}) {
  return (
    <article className="benefit-card">

      <div className="benefit-icon">
        {icon}
      </div>

      <h3>
        {title}
      </h3>

      <p>
        {description}
      </p>

    </article>
  );
}

export default App;