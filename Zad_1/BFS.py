from collections import deque
import finding_Steps


def solve_bfs(width, height, start_state, max_depth):
    goal_state = tuple(range(1, width * height)) + (0,)
    queue = deque([(start_state, [])])
    visited = set([start_state])

    while queue:
        current_state, path = queue.popleft()

        if current_state == goal_state:
            return path

        if len(path) >= max_depth:
            return None

        for neighbour, move in finding_Steps.find_possible_moves(width, height, current_state):
            if neighbour not in visited:
                queue.append((neighbour, path + [move]))
                visited.add(neighbour)

    return None




