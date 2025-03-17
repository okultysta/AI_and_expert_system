from collections import deque

def solve_bfs(width, height, start_state):
    goal_state = tuple(range(1, width*height)) + (0,)
    states_to_check = deque([start_state])
    visited = set([start_state])
    parent_map = {start_state: None}
    max_checked = 1000000000
    checked = 0

    while states_to_check and checked < max_checked:
        current_state = states_to_check.popleft()

        if current_state == goal_state:
            path = find_path(parent_map, current_state)
            moves = get_moves_sequence(path, width)
            return path, moves

        for neighbor in find_possible_moves(width, height, current_state):
            if neighbor not in visited:
                states_to_check.append(neighbor)
                visited.add(neighbor)
                parent_map[neighbor] = current_state
                checked += 1

    return None, None




def find_possible_moves(width, height, state):
    state = list(state)
    empty_filed_index = state.index(0)
    row, col = divmod(empty_filed_index, width)
    neighbors  = []

    moves = [(-1,0),(1,0),(0,1),(0,-1)]

    for horizontal, vertical in moves:
        new_row, new_col = row + horizontal, col + vertical
        if 0 <= new_row < height and 0 <= new_col < width:
            new_index = new_row * width + new_col
            new_state = state[:]
            new_state[new_index], new_state[empty_filed_index] = new_state[empty_filed_index], new_state[new_index]
            neighbors.append(tuple(new_state))

    return neighbors

def find_path(parent_map, state):
    path = []
    while state:
        path.append(state)
        state = parent_map[state]
    path.reverse()
    return path

def get_moves_sequence(path, width):
    moves = {
        (0, 1): "U",
        (0, -1): "D",
        (1, 0): "R",
        (-1, 0): "L",
    }
    moves_sequence = []
    for i in range(len(path)-1):
        current_state = path[i]
        next_state = path[i+1]

        current_index = current_state.index(0)
        next_index = next_state.index(0)

        row, col = divmod(current_index, width)
        next_row, next_col = divmod(next_index, width)

        move = (next_row - row, next_col - col)
        moves_sequence.append(moves[move])

    return moves_sequence