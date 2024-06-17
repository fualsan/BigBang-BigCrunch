
import jax.numpy as jnp
from jax import random


def calculate_center_of_mass(x, fitness):
    nom = jnp.sum((1/fitness)*x)
    denom = jnp.sum(1/fitness)
    x_c = nom / denom
    return x_c


def calculate_new_candidates(key, x, x_c, l, k):
    r = random.normal(key, x.shape)
    new_solutions = x_c + (l*r/k)
    return new_solutions


def find_solution(solutions, fitness_func, max_iterations, l, key):
    x_c_history = []
    fitness_history = []
    solutions_history = []

    best_fitness = float('-inf')
    best_x_c = None
    best_solutions = None

    ### MAIN LOOP ###
    #################
    for k in range(1, max_iterations+1):
        _fitness = fitness_func(solutions)
        _x_c = calculate_center_of_mass(solutions, _fitness)
        solutions = calculate_new_candidates(key, solutions, _x_c, l, k)

        solutions_history.append(solutions)

        for idx, _f in enumerate(_fitness):
            if _f > best_fitness:
                best_fitness = _f.copy()
                best_x_c = _x_c.copy()
                best_solutions = solutions[idx].copy()
            
        x_c_history.append(_x_c)
        fitness_history.append(_fitness.sum())

        print(f'Iteration: {k}, Fitness: {_fitness.sum():.4f}, CoM: {_x_c:.4f}')
        print(solutions)
        print('\n')
    #################
    #################

    print('Best fitness', best_fitness)
    print('Best CoM', best_x_c)
    print('Best solution:')
    print(best_solutions)
    print(f'f({best_solutions})={fitness_func(best_solutions)}')   

    return x_c_history, fitness_history, solutions_history
