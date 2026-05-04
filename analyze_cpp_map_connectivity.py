#
# Analyze whether randomly generated CPP maps are fully reachable from the
# initial agent position. This does not change the environment or the agent.
#
# Example:
#   python analyze_cpp_map_connectivity.py 10 12 400 1000
#

import argparse
from collections import deque

from gymnasium_env.grid_world_cpp import GridWorldCPPEnv


def reachable_free_cells(env):
    blocked = {tuple(obstacle) for obstacle in env.obstacles_locations}
    free_cells = {
        (x, y)
        for x in range(env.size)
        for y in range(env.size)
        if (x, y) not in blocked
    }

    start = tuple(env._agent_location)
    queue = deque([start])
    visited = {start}

    while queue:
        x, y = queue.popleft()
        for neighbor in ((x + 1, y), (x, y - 1), (x - 1, y), (x, y + 1)):
            if neighbor in free_cells and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return len(visited), len(free_cells)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dim", type=int)
    parser.add_argument("obstacles", type=int)
    parser.add_argument("max_steps", type=int)
    parser.add_argument("episodes", type=int, nargs="?", default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    env = GridWorldCPPEnv(
        size=args.dim,
        obs_quantity=args.obstacles,
        max_steps=args.max_steps,
        render_mode="rgb_array",
    )

    disconnected = 0
    reachable_ratios = []

    for _ in range(args.episodes):
        env.reset()
        reachable, total_free = reachable_free_cells(env)
        ratio = reachable / total_free
        reachable_ratios.append(ratio)
        if reachable < total_free:
            disconnected += 1

    print(f"Grid: {args.dim}x{args.dim}")
    print(f"Obstacles: {args.obstacles}")
    print(f"Episodes: {args.episodes}")
    print(f"Disconnected maps: {disconnected}/{args.episodes}")
    print(f"Average reachable free cells: {sum(reachable_ratios) / len(reachable_ratios):.4f}")
    print(f"Minimum reachable free cells: {min(reachable_ratios):.4f}")


if __name__ == "__main__":
    main()
