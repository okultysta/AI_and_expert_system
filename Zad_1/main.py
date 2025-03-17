
state = tuple(range(1, 6)) + (0,)

print(state)
state = list(state)
state[1], state[2] = state[2], state[1]
print(state)