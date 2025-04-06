import heapq
import time

import finding_Steps
import utils


def solve_heuristics(width, height, start_state, max_depth, heuristic, solution_filename, info_filename):
    start_time = time.time()
    goal_state = tuple(range(1, width * height)) + (0,)
    priority_queue = []
    heapq.heappush(priority_queue, (0, start_state, []))
    visited = {start_state: 0}

    while priority_queue:
        _, current_state, path = heapq.heappop(
            priority_queue)  # _ oznacza ignorowanie pierwszej wartości - priorytet nie jest nam do niczego potrzebny

        if current_state == goal_state:
            execution_time = (time.time() - start_time) * 1000
            utils.zapisz_rozwiazanie(solution_filename, len(path), path)
            utils.zapisz_informacje_dodatkowe(info_filename, len(path), len(visited), 0,
                                              0, execution_time)
            print("Czas wykonania:" + f"{execution_time:.3f}\n")
            return path

        if len(path) >= max_depth:
            utils.zapisz_rozwiazanie(solution_filename, -1, path)
            return None

        for neighbour, move in finding_Steps.find_possible_moves(width, height, current_state):
            new_cost = len(path) + 1
            if neighbour not in visited or new_cost < visited[neighbour]:
                visited[neighbour] = new_cost
                priority = new_cost + heuristic(neighbour, width, goal_state)
                heapq.heappush(priority_queue, (priority, neighbour, path + [move]))
    utils.zapisz_rozwiazanie(solution_filename, -1, path)
    return None


def calculate_hamming(state, width, goal_state):
    return sum(1 for i in range(len(state)) if state[i] != 0 and state[i] != goal_state[i])


def calculate_manhattan(state, width, goal_state):
    distance = 0
    for i in range(len(state)):
        if state[i] != 0:
            goal_index = goal_state.index(state[i])
            current_row, current_col = divmod(i, width)
            goal_row, goal_col = divmod(goal_index, width)
            distance += abs(current_row - goal_row) + abs(current_col - goal_col)
    return distance
