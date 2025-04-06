import time
from collections import deque
import finding_Steps
import utils


def solve_bfs(width, height, start_state, max_depth, search_order, solution_filename, info_filename):
    start_time = time.time()
    goal_state = tuple(range(1, width * height)) + (0,)
    queue = deque([(start_state, [])])
    visited = {start_state}

    while queue:
        current_state, path = queue.popleft()

        if current_state == goal_state:
            execution_time = (time.time() - start_time)*1000
            utils.zapisz_rozwiazanie(solution_filename, len(path), path)
            utils.zapisz_informacje_dodatkowe(info_filename, len(path),len(visited), 0,
                                              0, execution_time)
            print("Czas wykonania:" + f"{execution_time:.3f}\n")
            return path

        if len(path) >= max_depth:
            utils.zapisz_rozwiazanie(solution_filename, -1, path)
            return None

        for neighbour, move in finding_Steps.find_possible_moves(width, height, current_state, search_order):
            if neighbour not in visited:
                queue.append((neighbour, path + [move]))
                visited.add(neighbour)
    utils.zapisz_rozwiazanie(solution_filename, -1, None)
    return None




