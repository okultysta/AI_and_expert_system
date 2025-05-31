def find_possible_moves(width, height, state, search_order=None):
    # Jeśli nie podano kolejności, ustaw domyślną kolejność
    if search_order is None:
        search_order = ["U", "D", "R", "L"]

    state = list(state)
    empty_filed_index = state.index(0)
    row, col = divmod(empty_filed_index, width)
    neighbors = []

    # Mapa ruchów z ich współrzędnymi
    move_directions = {
        "U": (-1, 0),
        "D": (1, 0),
        "R": (0, 1),
        "L": (0, -1)
    }

    # Iteracja po ruchach w podanym porządku
    for move in search_order:
        horizontal, vertical = move_directions[move]
        new_row, new_col = row + horizontal, col + vertical
        if 0 <= new_row < height and 0 <= new_col < width:
            new_index = new_row * width + new_col
            new_state = state[:]
            new_state[new_index], new_state[empty_filed_index] = new_state[empty_filed_index], new_state[new_index]
            neighbors.append((tuple(new_state), move))

    return neighbors
