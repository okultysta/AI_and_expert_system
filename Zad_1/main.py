import random as rnd
from BFS import solve_bfs

numbers = list(range(0, 9))
rnd.shuffle(numbers)
state = tuple(numbers)

path, moves = solve_bfs(3,3,state)

if path is not None:
    for board in path:
        print(board)
    print()

print(state)
print(" ".join(moves))
