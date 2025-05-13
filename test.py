# rvo2_tests.py

import math
import random
import rvo2
import matplotlib.pyplot as plt

def normalize(dx, dy):
    """Normalize a vector (dx, dy) to unit length."""
    mag = math.hypot(dx, dy)
    return (dx/mag, dy/mag) if mag > 0 else (0.0, 0.0)

def set_preferred_velocities(sim, goals, jitter=1e-2):
    """Set each agent's preferred velocity toward its goal plus jitter."""
    two_pi = 2 * math.pi
    for i, (gx, gy) in enumerate(goals):
        px, py = sim.getAgentPosition(i)
        dx, dy = gx - px, gy - py

        # C++ Blocks logic: normalize if far, else use raw
        if dx*dx + dy*dy > 1.0:
            vx, vy = normalize(dx, dy)
        else:
            vx, vy = dx, dy

        # add small jitter to break symmetry
        angle = random.random() * two_pi
        r = random.random() * jitter
        vx += math.cos(angle) * r
        vy += math.sin(angle) * r

        sim.setAgentPrefVelocity(i, (vx, vy))

def simulate_square(
    side_length,
    delta_range=(0.01, 0.03),
    jitter=1e-2,
    neighborDist=50,
    maxNeighbors=5,
    timeHorizon=5.0,
    timeHorizonObst=2.0,
    radius=0.075,
    maxSpeed=0.3,
    max_steps=100000,
    goal_eps=0.02
):
    """Simulate 4 agents on a square of given side_length."""
    # Build simulator
    sim = rvo2.PyRVOSimulator(
        1/60.0, neighborDist, maxNeighbors,
        timeHorizon, timeHorizonObst,
        radius, maxSpeed
    )

    # Jittered corner offsets
    b1, b2 = delta_range
    starts = [
        (random.uniform(b1, b2), random.uniform(b1, b2)),
        (random.uniform(b1, b2), side_length - random.uniform(b1, b2)),
        (side_length - random.uniform(b1, b2), side_length - random.uniform(b1, b2)),
        (side_length - random.uniform(b1, b2), random.uniform(b1, b2)),
    ]
    goals = [
        (side_length - random.uniform(b1, b2), side_length - random.uniform(b1, b2)),
        (side_length - random.uniform(b1, b2), random.uniform(b1, b2)),
        (random.uniform(b1, b2), random.uniform(b1, b2)),
        (random.uniform(b1, b2), side_length - random.uniform(b1, b2)),
    ]

    # Add agents
    for pos in starts:
        sim.addAgent(pos)

    trajectories = [[] for _ in starts]

    # Main loop
    for _ in range(max_steps):
        set_preferred_velocities(sim, goals, jitter)
        for i in range(len(starts)):
            trajectories[i].append(sim.getAgentPosition(i))
        sim.doStep()
        if all(
            math.hypot(sim.getAgentPosition(i)[0] - goals[i][0],
                       sim.getAgentPosition(i)[1] - goals[i][1]) < goal_eps
            for i in range(len(starts))
        ):
            break

    return starts, goals, trajectories

def simulate_circle(
    num_agents,
    circle_radius,
    jitter=1e-2,
    neighborDist=50,
    maxNeighbors=5,
    timeHorizon=5.0,
    timeHorizonObst=2.0,
    radius=0.075,
    maxSpeed=0.3,
    max_steps=100000,
    goal_eps=0.02
):
    """Simulate num_agents placed on a circle moving to opposite points."""
    sim = rvo2.PyRVOSimulator(
        1/60.0, neighborDist, maxNeighbors,
        timeHorizon, timeHorizonObst,
        radius, maxSpeed
    )

    # Generate starts and goals
    starts, goals = [], []
    for i in range(num_agents):
        theta = 2 * math.pi * i / num_agents
        x = circle_radius * math.cos(theta)
        y = circle_radius * math.sin(theta)
        # add slight position jitter
        dx = random.uniform(-jitter, jitter)
        dy = random.uniform(-jitter, jitter)
        starts.append((x + dx, y + dy))

        # opposite angle
        opp_theta = theta + math.pi
        gx = circle_radius * math.cos(opp_theta) + random.uniform(-jitter, jitter)
        gy = circle_radius * math.sin(opp_theta) + random.uniform(-jitter, jitter)
        goals.append((gx, gy))

    # Add agents
    for pos in starts:
        sim.addAgent(pos)

    trajectories = [[] for _ in starts]

    # Main loop
    for _ in range(max_steps):
        set_preferred_velocities(sim, goals, jitter)
        for i in range(len(starts)):
            trajectories[i].append(sim.getAgentPosition(i))
        sim.doStep()
        if all(
            math.hypot(sim.getAgentPosition(i)[0] - goals[i][0],
                       sim.getAgentPosition(i)[1] - goals[i][1]) < goal_eps
            for i in range(len(starts))
        ):
            break

    return starts, goals, trajectories

def plot_trajectories(starts, goals, trajectories, title="Trajectories"):

    for i, traj in enumerate(trajectories):
        xs, ys = zip(*traj)
        plt.plot(xs, ys, label=f'Agent {i}')
        plt.scatter(*starts[i], marker='o')
        plt.scatter(*goals[i], marker='x', s=100)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Example usage:
    # Square test
    L = 1.0
    starts_sq, goals_sq, trajs_sq = simulate_square(side_length=L)
    plot_trajectories(starts_sq, goals_sq, trajs_sq, title=f"Square L={L}")

    # Circle test
    N = 8
    R = 1.0
    starts_circ, goals_circ, trajs_circ = simulate_circle(num_agents=N, circle_radius=R)
    plot_trajectories(starts_circ, goals_circ, trajs_circ, title=f"Circle N={N}, R={R}")
