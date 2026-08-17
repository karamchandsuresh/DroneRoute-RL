import { useEffect, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [environment, setEnvironment] = useState(null);
  const [routeData, setRouteData] = useState(null);
  const [dronePosition, setDronePosition] = useState([0, 0]);
  const [algorithm, setAlgorithm] = useState("q-learning");

  const [loading, setLoading] = useState(false);
  const [isAnimating, setIsAnimating] = useState(false);
  const [error, setError] = useState("");

  // Exploration demo state
  const [demoEvent, setDemoEvent] = useState(null);
  const [demoRunning, setDemoRunning] = useState(false);
  const [attemptedPosition, setAttemptedPosition] = useState(null);

  useEffect(() => {
    fetchEnvironment();
  }, []);

  const fetchEnvironment = async () => {
    try {
      const response = await fetch(`${API_URL}/environment`);

      if (!response.ok) {
        throw new Error("Failed to load environment.");
      }

      const data = await response.json();

      setEnvironment(data);
      setDronePosition(data.start);
    } catch {
      setError(
        "Could not connect to the DroneRoute RL backend."
      );
    }
  };

  const sleep = (milliseconds) => {
    return new Promise((resolve) =>
      setTimeout(resolve, milliseconds)
    );
  };

  const getAlgorithmLabel = () => {
    return algorithm === "q-learning"
      ? "Q-Learning"
      : "DQN";
  };

  // --------------------------------------------------
  // Run Q-Learning / DQN
  // --------------------------------------------------

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
        `${API_URL}${endpoint}`
      );

      if (!response.ok) {
        throw new Error(
          `${getAlgorithmLabel()} route request failed.`
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
    } catch {
      setLoading(false);
      setIsAnimating(false);

      setError(
        `Unable to generate the ${getAlgorithmLabel()} route.`
      );
    }
  };

  // --------------------------------------------------
  // Exploration / Reward Demo
  // --------------------------------------------------

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
      setDronePosition(environment.start);

      const response = await fetch(
        `${API_URL}/demo/exploration`
      );

      if (!response.ok) {
        throw new Error(
          "Exploration demo request failed."
        );
      }

      const data = await response.json();

      for (const event of data.events) {
        // Start position for the current event
        setDronePosition(event.from);

        setDemoEvent({
          ...event,
          phase: "action",
        });

        await sleep(900);

        // Show the attempted location
        setAttemptedPosition(event.attempted);

        await sleep(700);

        // Show the actual resulting position
        setDronePosition(event.to);

        setDemoEvent({
          ...event,
          phase: "result",
        });

        await sleep(1500);

        setAttemptedPosition(null);
      }

      setDemoRunning(false);
    } catch {
      setDemoRunning(false);
      setAttemptedPosition(null);

      setError(
        "Unable to run the exploration demo."
      );
    }
  };

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
      setDronePosition(environment.start);
    }
  };

  // --------------------------------------------------
  // Grid helpers
  // --------------------------------------------------

  const isSamePosition = (first, second) => {
    return (
      first &&
      second &&
      first[0] === second[0] &&
      first[1] === second[1]
    );
  };

  const isObstacle = (row, col) => {
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

  const isAttemptedCell = (row, col) => {
    if (!attemptedPosition) {
      return false;
    }

    return (
      attemptedPosition[0] === row &&
      attemptedPosition[1] === col
    );
  };

  const getCellClass = (row, col) => {
    const classes = ["grid-cell"];

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

    if (isAttemptedCell(row, col)) {
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

  const getCellContent = (row, col) => {
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

  // --------------------------------------------------
  // Exploration helpers
  // --------------------------------------------------

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

  // --------------------------------------------------
  // Metric helpers
  // --------------------------------------------------

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
      return `${routeData.battery_used}/${environment.max_battery}`;
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
    if (demoEvent && !routeData) {
      return "Cumulative Reward";
    }

    return "Reward";
  };

  // --------------------------------------------------
  // Loading screen
  // --------------------------------------------------

  if (!environment) {
    return (
      <div className="loading-screen">
        <h2>DroneRoute RL</h2>

        <p>
          {error || "Loading environment..."}
        </p>
      </div>
    );
  }

  return (
    <div className="app">
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

      <main className="dashboard">
        {/* ============================================
            Environment
        ============================================ */}

        <section className="panel environment-panel">
          <div className="panel-heading">
            <div>
              <p className="section-label">
                ENVIRONMENT
              </p>

              <h2>Delivery Grid</h2>
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
              { length: environment.grid_size },
              (_, row) =>
                Array.from(
                  {
                    length:
                      environment.grid_size,
                  },
                  (_, col) => (
                    <div
                      className={getCellClass(
                        row,
                        col
                      )}
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
            <span>✕ Obstacle</span>
          </div>
        </section>

        {/* ============================================
            RL Controls
        ============================================ */}

        <section className="panel control-panel">
          <p className="section-label">
            RL AGENT
          </p>

          <h2>{getAlgorithmLabel()}</h2>

          <p className="description">
            Select an RL algorithm, train the
            agent, and visualize the learned
            battery-aware delivery route.
          </p>

          <div className="algorithm-control">
            <label htmlFor="algorithm">
              Algorithm
            </label>

            <select
              id="algorithm"
              value={algorithm}
              onChange={handleAlgorithmChange}
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
              DQN uses a neural network and may
              take longer to train on CPU.
            </p>
          )}

          {/* ============================================
              Exploration Demo
          ============================================ */}

          <div className="demo-section">
            <div className="demo-divider" />

            <p className="section-label">
              LEARNING DEMONSTRATION
            </p>

            <h3>Reward & Penalty Exploration</h3>

            <p className="demo-description">
              See how the environment responds
              when the drone moves normally,
              hits an obstacle, crosses a
              boundary, or reaches the goal.
            </p>

            <button
              className="demo-button"
              onClick={runExplorationDemo}
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
                  <strong>Step:</strong>{" "}
                  {demoEvent.step}
                </p>

                <p>
                  <strong>Action:</strong>{" "}
                  {demoEvent.action}
                </p>

                <p>
                  <strong>From:</strong>{" "}
                  ({demoEvent.from[0]},{" "}
                  {demoEvent.from[1]})
                </p>

                <p>
                  <strong>Attempted:</strong>{" "}
                  ({demoEvent.attempted[0]},{" "}
                  {demoEvent.attempted[1]})
                </p>

                <p>
                  <strong>Result:</strong>{" "}
                  ({demoEvent.to[0]},{" "}
                  {demoEvent.to[1]})
                </p>

                <p>
                  <strong>
                    Cumulative Reward:
                  </strong>{" "}
                  {demoEvent.cumulative_reward}
                </p>

                <p>
                  <strong>
                    Battery Remaining:
                  </strong>{" "}
                  {demoEvent.battery_remaining}/
                  {environment.max_battery}
                </p>

                <p className="event-message">
                  {demoEvent.message}
                </p>
              </div>
            )}
          </div>

          {error && (
            <p className="error-message">
              {error}
            </p>
          )}

          {/* ============================================
              Metrics
          ============================================ */}

          <div className="metrics">
            <Metric
              title="Steps"
              value={getStepsValue()}
            />

            <Metric
              title={getRewardMetricTitle()}
              value={getRewardValue()}
            />

            <Metric
              title="Battery Used"
              value={getBatteryUsedValue()}
            />

            <Metric
              title="Success"
              value={getSuccessValue()}
            />
          </div>

          {/* ============================================
              Learned Route Information
          ============================================ */}

          {routeData && (
            <div className="route-details">
              <h3>Algorithm</h3>

              <p>{routeData.algorithm}</p>

              <h3>Learned Actions</h3>

              <p>
                {routeData.actions.join(" → ")}
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
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

function Metric({ title, value }) {
  return (
    <div className="metric-card">
      <span>{title}</span>
      <strong>{value}</strong>
    </div>
  );
}

export default App;