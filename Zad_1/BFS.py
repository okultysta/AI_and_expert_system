from collections import deque

def solve_fbs(width, height, start_state):
    goal_state = tuple(range(1, width*height)) + (0,)
    states_to_check = deque([start_state])
    checked = set([start_state])
    parent_map = {start_state: 0}

    while states_to_check:
        current_state = states_to_check.popleft()

        if current_state == goal_state:
            return current_state

        for neighbor in find_possible_moves(width, height, current_state):
            if neighbor not in checked:
                states_to_check.append(neighbor)
                checked.add(neighbor)
                parent_map[neighbor] = current_state

    return None




def find_possible_moves(width, height, state):
    state = list(state)
    empty_filed_index = state.index(0)
    row, col = divmod(empty_filed_index, width)
    neighbors  = []

    moves = [(-1,0),(1,0),(0,1),(0,-1)]

    for horizontal, vertical in moves:
        new_row, new_col = row + horizontal, col + vertical
        if 0 <= new_row < width and 0 <= new_col < height:
            new_index = new_row * width + new_col
            new_state = state[:]
            new_state[new_index], new_state[empty_filed_index] = new_state[empty_filed_index], new_state[new_index]
            neighbors.append(new_state)

    return neighbors