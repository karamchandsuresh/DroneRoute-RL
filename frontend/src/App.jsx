import { useEffect, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

const SCENARIOS = {
  standard: {
    label: "Standard Delivery",
    shortDescription:
      "Normal last-mile delivery with moderate obstacles.",
  },

  urban: {
    label: "Urban Restricted-Zone Delivery",
    shortDescription:
      "Delivery through an urban area containing additional restricted zones.",
  },

  low_battery: {
    label: "Low-Battery Delivery",
    shortDescription:
      "Energy-constrained delivery with limited battery capacity.",
  },
};

function App() {
  const [environment, setEnvironment] = useState(null);
  const [routeData, setRouteData] = useState(null);

  const [dronePosition, setDronePosition] = useState([
    0, 0,
  ]);

  const [algorithm, setAlgorithm] =
    useState("q-learning");

  const [scenario, setScenario] =
    useState("standard");

  const [loading, setLoading] = useState(false);

  const [isAnimating, setIsAnimating] =
    useState(false);

  const [error, setError] = useState("");

  // Exploration demo
  const [demoEvent, setDemoEvent] =
    useState(null);

  const [demoRunning, setDemoRunning] =
    useState(false);

  const [
    attemptedPosition,
    setAttemptedPosition,
  ] = useState(null);

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

      // Prevent undefined or invalid scenarios
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

      const data = await response.json();

      setEnvironment(data);
      setDronePosition(data.start);
      setRouteData(null);
      setDemoEvent(null);
      setAttemptedPosition(null);
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
      setTimeout(resolve, milliseconds)
    );
  };

  const getAlgorithmLabel = () => {
    return algorithm === "q-learning"
      ? "Q-Learning"
      : "Deep Q-Network (DQN)";
  };

  // ==================================================
  // RUN RL AGENT
  // ==================================================

  const runAgent = async () => {
    if (
      !environment ||
      loading ||
      isAnimating ||
      demoRunning
    ) {
      return;
    }

    try {
      setLoading(true);
      setError("");
      setRouteData(null);
      setDemoEvent(null);
      setAttemptedPosition(null);

      setDronePosition(environment.start);

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

      const data = await response.json();

      setRouteData(data);
      setLoading(false);
      setIsAnimating(true);

      for (const position of data.route) {
        setDronePosition(position);

        await sleep(450);
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
  // EXPLORATION DEMO
  // ==================================================

  const runExplorationDemo = async () => {
    if (
      !environment ||
      loading ||
      isAnimating ||
      demoRunning
    ) {
      return;
    }

    try {
      setDemoRunning(true);
      setError("");
      setRouteData(null);
      setDemoEvent(null);
      setAttemptedPosition(null);

      /*
       * The exploration demonstration uses the
       * standard environment because the action
       * sequence was designed specifically to
       * demonstrate:
       *
       * - boundary penalty
       * - normal movement penalty
       * - obstacle collision penalty
       * - destination reward
       */

      const response = await fetch(
        `${API_URL}/demo/exploration`
      );

      if (!response.ok) {
        throw new Error(
          "Exploration demo request failed."
        );
      }

      const data = await response.json();

      /*
       * Temporarily show the Standard Delivery
       * environment while running the controlled
       * exploration demonstration.
       */

      if (scenario !== "standard") {
        const environmentResponse =
          await fetch(
            `${API_URL}/environment?scenario=standard`
          );

        if (!environmentResponse.ok) {
          throw new Error(
            "Failed to load exploration environment."
          );
        }

        const standardEnvironment =
          await environmentResponse.json();

        setEnvironment(
          standardEnvironment
        );

        setDronePosition(
          standardEnvironment.start
        );
      } else {
        setDronePosition(
          environment.start
        );
      }

      for (const event of data.events) {
        // Position before action
        setDronePosition(event.from);

        setDemoEvent({
          ...event,
          phase: "action",
        });

        await sleep(900);

        // Show attempted location
        setAttemptedPosition(
          event.attempted
        );

        await sleep(700);

        // Show actual resulting location
        setDronePosition(event.to);

        setDemoEvent({
          ...event,
          phase: "result",
        });

        await sleep(1500);

        setAttemptedPosition(null);
      }

      setDemoRunning(false);

      /*
       * Restore the scenario selected by the user
       * after the exploration demonstration.
       */

      if (scenario !== "standard") {
        await fetchEnvironment(
          scenario
        );
      }
    } catch (err) {
      console.error(err);

      setDemoRunning(false);
      setAttemptedPosition(null);

      setError(
        "Unable to run the exploration demo."
      );
    }
  };

  // ==================================================
  // SCENARIO CHANGE
  // ==================================================

  const handleScenarioChange = (
    newScenario
  ) => {
    if (
      loading ||
      isAnimating ||
      demoRunning
    ) {
      return;
    }

    /*
     * Scenario buttons directly pass:
     *
     * standard
     * urban
     * low_battery
     *
     * This avoids scenario=undefined.
     */

    if (
      !newScenario ||
      !Object.prototype.hasOwnProperty.call(
        SCENARIOS,
        newScenario
      )
    ) {
      return;
    }

    setScenario(newScenario);

    setRouteData(null);
    setDemoEvent(null);
    setAttemptedPosition(null);
    setError("");
  };

  // ==================================================
  // ALGORITHM CHANGE
  // ==================================================

  const handleAlgorithmChange = (event) => {
    if (
      loading ||
      isAnimating ||
      demoRunning
    ) {
      return;
    }

    setAlgorithm(event.target.value);

    setRouteData(null);
    setDemoEvent(null);
    setAttemptedPosition(null);

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

  const isObstacle = (row, col) => {
    if (!environment) {
      return false;
    }

    return environment.obstacles.some(
      ([obstacleRow, obstacleCol]) =>
        obstacleRow === row &&
        obstacleCol === col
    );
  };

  const isRouteCell = (row, col) => {
    if (!routeData) {
      return false;
    }

    return routeData.route.some(
      ([routeRow, routeCol]) =>
        routeRow === row &&
        routeCol === col
    );
  };

  const isAttemptedCell = (
    row,
    col
  ) => {
    if (!attemptedPosition) {
      return false;
    }

    return (
      attemptedPosition[0] === row &&
      attemptedPosition[1] === col
    );
  };

  const getCellClass = (
    row,
    col
  ) => {
    const classes = [
      "grid-cell",
    ];

    if (isObstacle(row, col)) {
      classes.push("obstacle");
    }

    if (
      isSamePosition(
        [row, col],
        environment.destination
      )
    ) {
      classes.push("destination");
    }

    if (
      isSamePosition(
        [row, col],
        environment.start
      )
    ) {
      classes.push("start");
    }

    if (isRouteCell(row, col)) {
      classes.push("route");
    }

    if (
      isAttemptedCell(row, col)
    ) {
      classes.push("attempted");
    }

    if (
      isSamePosition(
        [row, col],
        dronePosition
      )
    ) {
      classes.push("drone");
    }

    return classes.join(" ");
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

    if (isObstacle(row, col)) {
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
  // EXPLORATION HELPERS
  // ==================================================

  const getRewardClass = () => {
    if (!demoEvent) {
      return "";
    }

    if (demoEvent.reward > 0) {
      return "positive-reward";
    }

    if (demoEvent.reward <= -10) {
      return "negative-reward";
    }

    return "step-reward";
  };

  const getEventTitle = () => {
    if (!demoEvent) {
      return "";
    }

    switch (demoEvent.type) {
      case "obstacle":
        return "Obstacle Collision";

      case "boundary":
        return "Boundary Collision";

      case "goal":
        return "Destination Reached";

      default:
        return "Normal Movement";
    }
  };

  // ==================================================
  // METRICS
  // ==================================================

  const getStepsValue = () => {
    if (routeData) {
      return routeData.steps;
    }

    if (demoEvent) {
      return demoEvent.step;
    }

    return "--";
  };

  const getRewardValue = () => {
    if (routeData) {
      return routeData.total_reward;
    }

    if (demoEvent) {
      return demoEvent.cumulative_reward;
    }

    return "--";
  };

  const getBatteryUsedValue = () => {
    if (routeData) {
      return `${routeData.battery_used}/${
        routeData.battery_capacity ??
        environment.max_battery
      }`;
    }

    if (demoEvent) {
      return `${demoEvent.battery_used}/${environment.max_battery}`;
    }

    return "--";
  };

  const getSuccessValue = () => {
    if (routeData) {
      return routeData.destination_reached
        ? "Yes"
        : "No";
    }

    if (demoEvent) {
      return demoEvent.destination_reached
        ? "Yes"
        : "No";
    }

    return "--";
  };

  const getRewardMetricTitle = () => {
    if (
      demoEvent &&
      !routeData
    ) {
      return "Cumulative Reward";
    }

    return "Reward";
  };

  // ==================================================
  // LOADING SCREEN
  // ==================================================

  if (!environment) {
    return (
      <div className="loading-screen">
        <h2>DroneRoute RL</h2>

        <p>
          {error ||
            "Loading environment..."}
        </p>
      </div>
    );
  }

  // ==================================================
  // USER INTERFACE
  // ==================================================

  return (
    <div className="app">

      {/* =================================================
          HEADER
      ================================================= */}

      <header className="header">
        <div>
          <p className="eyebrow">
            REINFORCEMENT LEARNING PROJECT
          </p>

          <h1>DroneRoute RL</h1>

          <p className="subtitle">
            Drone Delivery Optimization using
            Reinforcement Learning
          </p>
        </div>

        <div className="status-badge">
          API Connected
        </div>
      </header>

      {/* =================================================
          SCENARIO SELECTOR
      ================================================= */}

      <section className="scenario-section">

        <div className="scenario-heading">

          <div>
            <p className="section-label">
              DELIVERY ENVIRONMENT
            </p>

            <h2>
              Choose Delivery Scenario
            </h2>
          </div>

          <span className="scenario-badge">
            {environment.name ||
              SCENARIOS[scenario].label}
          </span>

        </div>

        <p className="scenario-introduction">
          Test the reinforcement learning
          agents under different simulated
          delivery conditions.
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
                  demoRunning
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
            <span>Scenario</span>

            <strong>
              {environment.name ||
                SCENARIOS[scenario].label}
            </strong>
          </div>

          <div>
            <span>Battery</span>

            <strong>
              {environment.max_battery} units
            </strong>
          </div>

          <div>
            <span>Restricted Cells</span>

            <strong>
              {environment.obstacles.length}
            </strong>
          </div>

        </div>

        <p className="scenario-description">
          {environment.description ||
            SCENARIOS[scenario]
              .shortDescription}
        </p>

      </section>

      {/* =================================================
          SIMULATION DASHBOARD
      ================================================= */}

      <main className="dashboard">

        {/* ===============================================
            ENVIRONMENT GRID
        =============================================== */}

        <section className="panel environment-panel">

          <div className="panel-heading">

            <div>
              <p className="section-label">
                ENVIRONMENT
              </p>

              <h2>
                Delivery Grid
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
                      {getCellContent(
                        row,
                        col
                      )}
                    </div>
                  )
                )
            )}

          </div>

          <div className="legend">
            <span>🚁 Drone</span>
            <span>S Start</span>
            <span>G Goal</span>
            <span>✕ Restricted Zone</span>
          </div>

        </section>

        {/* ===============================================
            RL CONTROLS
        =============================================== */}

        <section className="panel control-panel">

          <p className="section-label">
            RL AGENT
          </p>

          <h2>
            {getAlgorithmLabel()}
          </h2>

          <p className="description">
            Select an RL algorithm,
            train the agent and visualize
            the battery-aware delivery
            route learned through interaction
            with the environment.
          </p>

          {/* Algorithm selector */}

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
                demoRunning
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

          {/* Run agent */}

          <button
            className="run-button"
            onClick={runAgent}
            disabled={
              loading ||
              isAnimating ||
              demoRunning
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
              experience replay and a target
              network. Training may take
              longer on CPU.
            </p>
          )}

          {/* =============================================
              EXPLORATION DEMONSTRATION
          ============================================= */}

          <div className="demo-section">

            <div className="demo-divider" />

            <p className="section-label">
              LEARNING DEMONSTRATION
            </p>

            <h3>
              Reward & Penalty Exploration
            </h3>

            <p className="demo-description">
              Visualize how rewards and
              penalties guide learning when
              the drone moves normally,
              encounters an obstacle, crosses
              a boundary or reaches the
              delivery destination.
            </p>

            <button
              className="demo-button"
              onClick={
                runExplorationDemo
              }
              disabled={
                loading ||
                isAnimating ||
                demoRunning
              }
            >
              {demoRunning
                ? "Running Exploration..."
                : "Show Exploration Demo"}
            </button>

            {demoEvent && (
              <div className="demo-event-card">

                <div className="demo-event-header">

                  <strong>
                    {getEventTitle()}
                  </strong>

                  <span
                    className={`reward-badge ${getRewardClass()}`}
                  >
                    Reward{" "}
                    {demoEvent.reward > 0
                      ? `+${demoEvent.reward}`
                      : demoEvent.reward}
                  </span>

                </div>

                <p>
                  <strong>
                    Step:
                  </strong>{" "}
                  {demoEvent.step}
                </p>

                <p>
                  <strong>
                    Action:
                  </strong>{" "}
                  {demoEvent.action}
                </p>

                <p>
                  <strong>
                    From:
                  </strong>{" "}
                  ({demoEvent.from[0]},{" "}
                  {demoEvent.from[1]})
                </p>

                <p>
                  <strong>
                    Attempted:
                  </strong>{" "}
                  (
                  {demoEvent.attempted[0]},
                  {" "}
                  {demoEvent.attempted[1]})
                </p>

                <p>
                  <strong>
                    Result:
                  </strong>{" "}
                  ({demoEvent.to[0]},{" "}
                  {demoEvent.to[1]})
                </p>

                <p>
                  <strong>
                    Cumulative Reward:
                  </strong>{" "}
                  {
                    demoEvent.cumulative_reward
                  }
                </p>

                <p>
                  <strong>
                    Battery Remaining:
                  </strong>{" "}
                  {
                    demoEvent.battery_remaining
                  }
                  /
                  {
                    environment.max_battery
                  }
                </p>

                <p className="event-message">
                  {demoEvent.message}
                </p>

              </div>
            )}

          </div>

          {/* Error */}

          {error && (
            <p className="error-message">
              {error}
            </p>
          )}

          {/* =============================================
              METRICS
          ============================================= */}

          <div className="metrics">

            <Metric
              title="Steps"
              value={getStepsValue()}
            />

            <Metric
              title={
                getRewardMetricTitle()
              }
              value={getRewardValue()}
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

          {/* =============================================
              ROUTE DETAILS
          ============================================= */}

          {routeData && (
            <div className="route-details">

              <h3>
                Algorithm
              </h3>

              <p>
                {routeData.algorithm}
              </p>

              {routeData.scenario_name && (
                <>
                  <h3>
                    Delivery Scenario
                  </h3>

                  <p>
                    {
                      routeData.scenario_name
                    }
                  </p>
                </>
              )}

              <h3>
                Learned Actions
              </h3>

              <p>
                {routeData.actions.join(
                  " → "
                )}
              </p>

              {routeData.training_success_rate !==
                undefined && (
                <>
                  <h3>
                    Training Success Rate
                  </h3>

                  <p>
                    {
                      routeData.training_success_rate
                    }
                    %
                  </p>
                </>
              )}

              {routeData.battery_failure_rate !==
                undefined && (
                <>
                  <h3>
                    Battery Failure Rate
                  </h3>

                  <p>
                    {
                      routeData.battery_failure_rate
                    }
                    %
                  </p>
                </>
              )}

              {routeData.average_reward_last_100 !==
                undefined && (
                <>
                  <h3>
                    Last 100 Average Reward
                  </h3>

                  <p>
                    {
                      routeData.average_reward_last_100
                    }
                  </p>
                </>
              )}

              {routeData.average_steps_last_100 !==
                undefined && (
                <>
                  <h3>
                    Last 100 Average Steps
                  </h3>

                  <p>
                    {
                      routeData.average_steps_last_100
                    }
                  </p>
                </>
              )}

            </div>
          )}

        </section>

      </main>

      {/* =================================================
          REAL-WORLD APPLICATION
      ================================================= */}

      <section className="real-world-section">

        <div className="section-intro">

          <p className="section-label">
            REAL-WORLD APPLICATION
          </p>

          <h2>
            Why DroneRoute RL Matters
          </h2>

          <p>
            DroneRoute RL demonstrates how
            reinforcement learning can support
            autonomous delivery decisions when
            a drone must consider route
            efficiency, restricted areas and
            limited battery capacity.
          </p>

        </div>

        {/* ===============================================
            PROBLEM AND SOLUTION
        =============================================== */}

        <div className="problem-solution-grid">

          <article className="info-card">

            <span className="info-number">
              01
            </span>

            <h3>
              Current Problem
            </h3>

            <p>
              Drone delivery routes can be
              affected by buildings,
              restricted zones, unnecessary
              movement and limited battery
              capacity. Inefficient decisions
              can increase energy usage or
              prevent successful delivery.
            </p>

          </article>

          <article className="info-card">

            <span className="info-number">
              02
            </span>

            <h3>
              Proposed Solution
            </h3>

            <p>
              DroneRoute RL uses reinforcement
              learning to allow an agent to
              learn navigation decisions from
              rewards and penalties instead of
              manually specifying every
              movement.
            </p>

          </article>

        </div>

        {/* ===============================================
            SIMULATION TO REAL WORLD
        =============================================== */}

        <div className="mapping-section">

          <div className="mapping-heading">

            <p className="section-label">
              SIMULATION TO REAL WORLD
            </p>

            <h2>
              What Does the Grid Represent?
            </h2>

          </div>

          <div className="mapping-grid">

            <MappingCard
              simulation="Grid"
              realWorld="Delivery Area / Map"
            />

            <MappingCard
              simulation="Start"
              realWorld="Warehouse / Dispatch Point"
            />

            <MappingCard
              simulation="Goal"
              realWorld="Customer Location"
            />

            <MappingCard
              simulation="Obstacles"
              realWorld="Buildings / No-Fly Zones"
            />

            <MappingCard
              simulation="Battery Units"
              realWorld="Available Drone Energy"
            />

            <MappingCard
              simulation="RL Actions"
              realWorld="Navigation Decisions"
            />

          </div>

        </div>

        {/* ===============================================
            OPERATIONAL FLOW
        =============================================== */}

        <div className="workflow-section">

          <p className="section-label">
            OPERATIONAL CONCEPT
          </p>

          <h2>
            How It Could Support a
            Delivery System
          </h2>

          <div className="workflow">

            <WorkflowStep
              number="1"
              title="Delivery Request"
              description="A customer delivery destination is received."
            />

            <div className="workflow-arrow">
              →
            </div>

            <WorkflowStep
              number="2"
              title="Environment Data"
              description="Route constraints, restricted areas and battery information are provided."
            />

            <div className="workflow-arrow">
              →
            </div>

            <WorkflowStep
              number="3"
              title="RL Decision"
              description="The trained agent evaluates the current state and selects navigation actions."
            />

            <div className="workflow-arrow">
              →
            </div>

            <WorkflowStep
              number="4"
              title="Delivery Route"
              description="The drone follows an efficient route toward the destination."
            />

          </div>

        </div>

        {/* ===============================================
            BENEFITS
        =============================================== */}

        <div className="benefits-section">

          <p className="section-label">
            PROJECT BENEFITS
          </p>

          <h2>
            What Is Being Optimized?
          </h2>

          <div className="benefits-grid">

            <BenefitCard
              icon="↗"
              title="Route Efficiency"
              description="Penalizing unnecessary movement encourages the agent to learn shorter and more efficient delivery behaviour."
            />

            <BenefitCard
              icon="⚡"
              title="Energy Awareness"
              description="Every attempted action consumes battery, connecting route efficiency with available delivery energy."
            />

            <BenefitCard
              icon="⊘"
              title="Restricted-Zone Avoidance"
              description="Large penalties discourage navigation through obstacles and simulated no-fly areas."
            />

            <BenefitCard
              icon="◎"
              title="Autonomous Decisions"
              description="The agent learns a navigation policy through interaction instead of following only manually programmed movements."
            />

          </div>

        </div>

        {/* ===============================================
            RL CONTRIBUTION
        =============================================== */}

        <div className="rl-explanation">

          <div>

            <p className="section-label">
              WHY REINFORCEMENT LEARNING?
            </p>

            <h2>
              Learning Through Consequences
            </h2>

            <p>
              The agent interacts with the
              delivery environment and receives
              feedback for its actions.
              Repeated interaction helps it
              learn which decisions produce
              better long-term outcomes.
            </p>

          </div>

          <div className="reward-system">

            <RewardRow
              label="Successful Delivery"
              value="+100"
              type="good"
            />

            <RewardRow
              label="Normal Movement"
              value="-1"
            />

            <RewardRow
              label="Boundary Violation"
              value="-10"
              type="bad"
            />

            <RewardRow
              label="Obstacle Collision"
              value="-20"
              type="bad"
            />

            <RewardRow
              label="Battery Depleted"
              value="-50"
              type="bad"
            />

          </div>

        </div>

        {/* ===============================================
            LIMITATIONS AND FUTURE SCOPE
        =============================================== */}

        <div className="limitations-grid">

          <article className="limitations-card">

            <p className="section-label">
              CURRENT LIMITATIONS
            </p>

            <h2>
              Prototype Scope
            </h2>

            <ul>

              <li>
                Uses a simplified 5 × 5
                simulated environment.
              </li>

              <li>
                Obstacles are predefined
                rather than detected from
                real sensors.
              </li>

              <li>
                Battery consumption is
                represented using simplified
                energy units.
              </li>

              <li>
                Weather, wind and payload
                weight are not currently
                modelled.
              </li>

              <li>
                The system does not currently
                control a physical drone.
              </li>

            </ul>

          </article>

          <article className="future-card">

            <p className="section-label">
              FUTURE SCOPE
            </p>

            <h2>
              From Simulation to Deployment
            </h2>

            <ul>

              <li>
                Integrate real GPS and
                geographical map data.
              </li>

              <li>
                Add dynamic obstacles and
                changing restricted zones.
              </li>

              <li>
                Incorporate weather and wind
                conditions.
              </li>

              <li>
                Use real drone battery
                telemetry and payload data.
              </li>

              <li>
                Connect the learned policy to
                a drone simulator or physical
                flight controller.
              </li>

            </ul>

          </article>

        </div>

        {/* ===============================================
            PROJECT SIGNIFICANCE
        =============================================== */}

        <div className="project-conclusion">

          <p className="section-label">
            PROJECT SIGNIFICANCE
          </p>

          <h2>
            A Foundation for Intelligent
            Drone Delivery
          </h2>

          <p>
            DroneRoute RL is a
            simulation-based prototype that
            demonstrates how reinforcement
            learning can be applied to
            autonomous, battery-aware delivery
            navigation. The current system
            provides a foundation for future
            integration with real maps,
            sensors, dynamic conditions and
            physical drones.
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

function MappingCard({
  simulation,
  realWorld,
}) {
  return (
    <div className="mapping-card">

      <span>
        {simulation}
      </span>

      <strong>
        →
      </strong>

      <p>
        {realWorld}
      </p>

    </div>
  );
}

function WorkflowStep({
  number,
  title,
  description,
}) {
  return (
    <div className="workflow-step">

      <span className="workflow-number">
        {number}
      </span>

      <h3>
        {title}
      </h3>

      <p>
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

function RewardRow({
  label,
  value,
  type = "",
}) {
  return (
    <div className="reward-row">

      <span>
        {label}
      </span>

      <strong
        className={
          type === "good"
            ? "reward-good"
            : type === "bad"
            ? "reward-bad"
            : ""
        }
      >
        {value}
      </strong>

    </div>
  );
}

export default App;