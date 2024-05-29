import heapq
import math

from posggym.envs.continuous.core import FloatCoord, Coord
from typing import Iterable, List, Tuple, Optional


def heuristic(a: FloatCoord, b: FloatCoord):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def is_collision(point: FloatCoord, obstacles: Iterable[Coord], obstacle_radius: float):
    return any((heuristic(point, x) <= obstacle_radius for x in obstacles))


def a_star_continuous(
    start: FloatCoord,
    goal: FloatCoord,
    obstacles: Iterable[FloatCoord],
    obstacle_radius: float,
    step_size: float = 0.5,
    goal_threshold: float = 0.1,
) -> Tuple[Optional[List[FloatCoord]], float]:
    open_set: List[Tuple[float, FloatCoord]] = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0.0}
    f_score = {start: heuristic(start, goal)}

    directions: List[FloatCoord] = [
        (step_size, 0),
        (0, step_size),
        (-step_size, 0),
        (0, -step_size),
        (step_size, step_size),
        (-step_size, -step_size),
        (step_size, -step_size),
        (-step_size, step_size),
    ]

    while open_set:
        current = heapq.heappop(open_set)[1]

        if heuristic(current, goal) < goal_threshold:
            path: List[FloatCoord] = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path = path[::-1]  # Return reversed path
            total_distance = sum(
                heuristic(path[i], path[i + 1]) for i in range(len(path) - 1)
            )
            total_distance += heuristic(path[-1], goal)  # Add final segment distance
            path.append(goal)  # Include goal in the path
            return path, total_distance

        for direction in directions:
            neighbor = (current[0] + direction[0], current[1] + direction[1])
            if not is_collision(neighbor, obstacles, obstacle_radius):
                tentative_g_score = g_score[current] + heuristic(current, neighbor)

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None, float("inf")  # No path found


# Example usage:
if __name__ == "__main__":
    start = (2.5, 2.5)
    goal = (4.75, 4.75)
    obstacles: List[FloatCoord] = [
        (2.5, 2.5),
        (3.0, 3.0),
        (3.5, 3.5),
    ]  # List of obstacle centers
    obstacle_radius = 0.5  # Radius of obstacles
    path, total_distance = a_star_continuous(start, goal, obstacles, obstacle_radius)
    print("Path:", path)
    print("Total Distance:", total_distance)
