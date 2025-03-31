import random as rnd
from BFS import solve_bfs
from DFS import solve_dfs
import heuristics

height = 4
width = 4
max_depth = 30

numbers = list(range(0, height*width))
rnd.shuffle(numbers)
#state = tuple(numbers)

state = (5, 1, 3, 4,
         9, 2, 6, 8,
         13, 14, 7, 11,
         0, 15, 10, 12)

state = tuple(state)


moves_dfs = solve_dfs(width,height,state, max_depth)

if moves_dfs is not None:
    print("(DFS): Solution found in", len(moves_dfs))
    print(" ".join(moves_dfs))
else:
    print("Combination not found(DFS)")


moves_bfs = solve_bfs(width,height,state, max_depth)

if moves_bfs is not None:
    print("(BFS): Solution found in", len(moves_bfs))
    print(" ".join(moves_bfs))
else:
    print("Combination not found(BFS)")


moves_hamming = heuristics.solve_heuristics(width, height, state, max_depth, heuristics.calculate_hamming)

if moves_hamming is not None:
    print("(Hamming): Solution found in", len(moves_hamming))
    print(" ".join(moves_hamming))
else:
    print("Combination not found(Hamming)")


moves_manhattan = heuristics.solve_heuristics(width, height, state, max_depth, heuristics.calculate_manhattan)

if moves_manhattan is not None:
    print("(Manhattan): Solution found in", len(moves_manhattan))
    print(" ".join(moves_manhattan))
else:
    print("Combination not found(Manhattan)")