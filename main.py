import jax
from jax import random
import jax.numpy as jnp

import matplotlib.pyplot as plt
from utilities import animate_solutions
from bb_bc import find_solution

plt.rcParams['axes.grid'] = True


print(f'JAX version: {jax.__version__}')
print(f'JAX local device count: {jax.local_device_count()}')
print(f'JAX process count: {jax.process_count()}')
print(f'JAX device: {jax.devices()}')


# for reproducibility
KEY = random.key(1234)

# number of candidate solutions
N = 10
  
# upper limit of the parameter
L = 1.0

# epochs
MAX_ITERATIONS = 500


def poly_fitness(x):
    return -(x-2.0)**2


def sigmoid_fitness(x):
    return (1 / (1 + jnp.exp(-x)))


if __name__ == '__main__':
        
    candidate_solutions = random.normal(KEY, (N,))

    x_c_history, fitness_history, solutions_history = find_solution(
        solutions=candidate_solutions, 
        fitness_func=poly_fitness, 
        #fitness_func=sigmoid_fitness,
        max_iterations=MAX_ITERATIONS, 
        l=L,
        key=KEY,
    )

    # stack on first dim, add dummy last dim
    solutions_history = jnp.stack(solutions_history)[..., None]

    # repeat last dim twice, 1D -> 2D
    animate_fig = animate_solutions(solutions_history.repeat(3, 2))

    animate_fig.save('solutions_animated.mp4')
    animate_fig.save('solutions_animated.gif')

    plt.scatter(list(range(len(x_c_history))), x_c_history, label='CoM')
    plt.scatter(list(range(len(fitness_history))), fitness_history, label='Fitness')
    plt.legend()
    plt.title('History')
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.show()
