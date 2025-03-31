import finding_Steps


def solve_dfs(width, height, start_state, max_depth):
    goal_state = tuple(range(1, width * height)) + (0,)
    stack = [(start_state, [])]
    #visited = set([start_state])

    while stack:
        current_state, path = stack.pop()

        if current_state == goal_state:
            return path

        if len(path) >= max_depth:
            continue

        for neighbour, move in finding_Steps.find_possible_moves(width, height, current_state):
            #if neighbour not in visited:
                stack.append((neighbour, path + [move]))
                #visited.add(neighbour)

    return None