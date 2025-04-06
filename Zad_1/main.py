import random as rnd
import sys
import itertools
from BFS import solve_bfs
from DFS import solve_dfs
import heuristics

import sys

args = sys.argv
if len(args) != 6:
    raise ValueError("Nieprawidłowa liczba argumentów! Powinno być 6 argumentów.")

algorithm = args[1]
add_param = args[2]
puzzle_file = args[3]
solution_file = args[4]
info_file = args[5]

height = 4
width = 4
max_depth = 30

numbers = list(range(0, height * width))
rnd.shuffle(numbers)
# state = tuple(numbers)

state = (5, 1, 3, 4,
         9, 2, 6, 8,
         13, 14, 7, 11,
         0, 15, 10, 12)

state = tuple(state)

# moves_dfs = solve_dfs(width, height, state, max_depth)
#
# if moves_dfs is not None:
#     print("(DFS): Solution found in", len(moves_dfs))
#     print(" ".join(moves_dfs))
# else:
#     print("Combination not found(DFS)")
#
# moves_bfs = solve_bfs(width, height, state, max_depth)
#
# if moves_bfs is not None:
#     print("(BFS): Solution found in", len(moves_bfs))
#     print(" ".join(moves_bfs))
# else:
#     print("Combination not found(BFS)")
#
# moves_hamming = heuristics.solve_heuristics(width, height, state, max_depth, heuristics.calculate_hamming)
#
# if moves_hamming is not None:
#     print("(Hamming): Solution found in", len(moves_hamming))
#     print(" ".join(moves_hamming))
# else:
#     print("Combination not found(Hamming)")
#
# moves_manhattan = heuristics.solve_heuristics(width, height, state, max_depth, heuristics.calculate_manhattan)
#
# if moves_manhattan is not None:
#     print("(Manhattan): Solution found in", len(moves_manhattan))
#     print(" ".join(moves_manhattan))
# else:
#     print("Combination not found(Manhattan)")


litery_male = 'rdul'
litery_duze = 'RDUL'

permutacje_male = [''.join(p) for p in itertools.permutations(litery_male, 4)]
permutacje_duze = [''.join(p) for p in itertools.permutations(litery_duze, 4)]


if  algorithm == "dfs":
    if add_param not in permutacje_male and add_param not in permutacje_duze:
        raise ValueError("Błędny schemat przeszukiwania sąsiadów!")
    moves_dfs = solve_dfs(width, height, state, max_depth, add_param, solution_file, info_file)
    if moves_dfs is not None:
        print("(DFS): Solution found in", len(moves_dfs))
        print(" ".join(moves_dfs))
    else:
        print("Combination not found (DFS)")

if algorithm == "bfs":
    if add_param not in permutacje_male and add_param not in permutacje_duze:
        raise ValueError("Błędny schemat przeszukiwania sąsiadów!")
    moves_bfs = solve_bfs(width, height, state, max_depth, add_param, solution_file, info_file)
    if moves_bfs is not None:
        print("(BFS): Solution found in", len(moves_bfs))
        print(" ".join(moves_bfs))
    else:
        print("Combination not found (BFS)")
if algorithm == "astr":
    if add_param == "hamm":
        moves_hamming = heuristics.solve_heuristics(width, height, state, max_depth, heuristics.calculate_hamming, solution_file, info_file)
        if moves_hamming is not None:
            print("(Hamming): Solution found in", len(moves_hamming))
            print(" ".join(moves_hamming))
        else:
            print("Combination not found(Hamming)")
    elif add_param == "manh":
        moves_manhattan = heuristics.solve_heuristics(width, height, state, max_depth, heuristics.calculate_manhattan, solution_file, info_file)
        if moves_manhattan is not None:
            print("(Manhattan): Solution found in", len(moves_manhattan))
            print(" ".join(moves_manhattan))
        else:
            print("Combination not found(Manhattan)")
    else:
        raise ValueError("Niepoprawny drugi argument wejsciowy!")
