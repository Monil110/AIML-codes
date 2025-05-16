import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return -x**2

def hill_climbing(initial_x, max_iterations=1000, step_size=0.9):
    current_x = initial_x
    current_value = f(current_x)
    x_vals = [current_x]
    y_vals = [current_value]
    for _ in range(max_iterations):
        neighbor_x = current_x + step_size * np.random.uniform(-1, 1)
        neighbor_value = f(neighbor_x)
        if neighbor_value > current_value:
            current_x = neighbor_x
            current_value = neighbor_value
            x_vals.append(current_x)
            y_vals.append(current_value)
    return current_x, current_value, x_vals, y_vals

initial_x = float(input("Enter the initial value for x: "))
best_x, best_y, x_list, y_list = hill_climbing(initial_x)

print(f"x = {best_x}")
print(f"f(x) = {best_y}")

gx = np.linspace(min(x_list), max(x_list), 1000)
gy = [f(x) for x in gx]

plt.plot(gx, gy)
plt.plot(x_list, y_list)
plt.title("Hill Climbing Algorithm")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()
