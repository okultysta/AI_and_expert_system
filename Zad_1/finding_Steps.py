def find_possible_moves(width, height, state):
    state = list(state)
    empty_filed_index = state.index(0)
    row, col = divmod(empty_filed_index, width)
    neighbors = []

    moves = [(-1, 0, "U"), (1, 0, "D"), (0, 1, "R"), (0, -1, "L")]

    for horizontal, vertical, move in moves:
        new_row, new_col = row + horizontal, col + vertical
        if 0 <= new_row < height and 0 <= new_col < width:
            new_index = new_row * width + new_col
            new_state = state[:]
            new_state[new_index], new_state[empty_filed_index] = new_state[empty_filed_index], new_state[new_index]
            neighbors.append((tuple(new_state), move))

    return neighbors