def print_state(state):
    for row in state:
        print(" ".join(map(str, row)))
    print()

def find_blank(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return i, j

def move(state, direction):
    i, j = find_blank(state)
    new_state = [row[:] for row in state]
    if direction == "up" and i > 0:
        new_state[i][j], new_state[i-1][j] = new_state[i-1][j], new_state[i][j]
        return new_state
    if direction == "down" and i < 2:
        new_state[i][j], new_state[i+1][j] = new_state[i+1][j], new_state[i][j]
        return new_state
    if direction == "left" and j > 0:
        new_state[i][j], new_state[i][j-1] = new_state[i][j-1], new_state[i][j]
        return new_state
    if direction == "right" and j < 2:
        new_state[i][j], new_state[i][j+1] = new_state[i][j+1], new_state[i][j]
        return new_state
    return None

def heuristic(state, goal):
    return sum(state[i][j] != goal[i][j] for i in range(3) for j in range(3))

def a_star(start, goal):
    OPEN = [[heuristic(start, goal), 0, start]]
    CLOSED = []

    while OPEN:
        OPEN.sort()
        f, g, current = OPEN.pop(0)
        print_state(current)
        if current == goal:
            print("Solution found!")
            return
        CLOSED.append(current)
        for d in ["up", "down", "left", "right"]:
            nxt = move(current, d)
            if nxt and nxt not in CLOSED:
                h = heuristic(nxt, goal)
                OPEN.append([g + 1 + h, g + 1, nxt])
    print("Solution not found")

initial_state = [
    [1, 2, 3],
    [8, 0, 4],
    [7, 6, 5]
]

goal_state = [
    [1, 2, 0],
    [8, 6, 3],
    [7, 5, 4]
]

a_star(initial_state, goal_state)
