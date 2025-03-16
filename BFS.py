from collections import deque

def solve_fbs(width, height, start_state):
    goal_state = tuple(range(1, width*height)) + (0,)
    states_to_check = deque([start_state])
    checked = set([start_state])
    parent_map = {start_state: 0}

    return