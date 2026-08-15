from environment.drone_env import DroneEnvironment


def main():
    env = DroneEnvironment()

    print("=== DroneRoute RL Environment Test ===")

    # -----------------------------
    # Test 1: Normal movement
    # -----------------------------
    env.reset()

    state, reward, done = env.step(3)  # RIGHT

    print("\nTest 1 - Normal Movement")
    print("State:", state)
    print("Reward:", reward)
    print("Done:", done)

    # -----------------------------
    # Test 2: Obstacle collision
    # -----------------------------
    state, reward, done = env.step(1)  # DOWN into (1,1)

    print("\nTest 2 - Obstacle Collision")
    print("State:", state)
    print("Reward:", reward)
    print("Done:", done)

    # -----------------------------
    # Test 3: Boundary collision
    # -----------------------------
    env.reset()

    state, reward, done = env.step(0)  # UP from (0,0)

    print("\nTest 3 - Boundary Collision")
    print("State:", state)
    print("Reward:", reward)
    print("Done:", done)

    # -----------------------------
    # Test 4: Destination
    # -----------------------------
    env.drone_position = (4, 3)

    state, reward, done = env.step(3)  # RIGHT into destination

    print("\nTest 4 - Destination Reached")
    print("State:", state)
    print("Reward:", reward)
    print("Done:", done)

    print("\nFinal Environment:")
    env.render()


if __name__ == "__main__":
    main()