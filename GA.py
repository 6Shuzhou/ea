from typing import Tuple 
import numpy as np

import ioh
from ioh import get_problem, logger, ProblemClass

budget = 100000

# To make results reproducible
np.random.seed(42)

def studentnumber1_studentnumber2_GA(problem: ioh.problem.PBO) -> None:
    population_size = 500  # Adjust based on preference
    mutation_rate = 1 / problem.meta_data.n_variables  # Inverse of problem dimension
    crossover_rate = 0.9  # Probability of crossover

    # Step 1: Initialize population
    population = np.random.randint(2, size=(population_size, problem.meta_data.n_variables))
    
    # Step 2: Evaluate initial population
    fitness = np.array([problem(ind) for ind in population])
    
    while problem.state.evaluations < budget:
        # Step 3: Selection (Tournament Selection)
        selected_indices = np.random.choice(population_size, size=population_size, replace=True)
        parents = population[selected_indices]

        # Step 4: Crossover
        offspring = []
        for i in range(0, population_size, 2):
            if np.random.rand() < crossover_rate and i + 1 < population_size:
                point = np.random.randint(1, problem.meta_data.n_variables)
                parent1, parent2 = parents[i], parents[i + 1]
                child1 = np.concatenate((parent1[:point], parent2[point:]))
                child2 = np.concatenate((parent2[:point], parent1[point:]))
                offspring.extend([child1, child2])
            else:
                offspring.extend([parents[i], parents[i + 1]])
        
        offspring = np.array(offspring[:population_size])  # Ensure size consistency
        
        # Step 5: Mutation
        for i in range(population_size):
            if np.random.rand() < mutation_rate:
                mutation_point = np.random.randint(problem.meta_data.n_variables)
                offspring[i][mutation_point] = 1 - offspring[i][mutation_point]  # Flip the bit
        
        # Step 6: Evaluate offspring
        offspring_fitness = np.array([problem(ind) for ind in offspring])
        
        # Step 7: Replace population (Elitism)
        combined_population = np.vstack((population, offspring))
        combined_fitness = np.hstack((fitness, offspring_fitness))
        best_indices = np.argsort(combined_fitness)[-population_size:]
        population = combined_population[best_indices]
        fitness = combined_fitness[best_indices]


def create_problem(dimension: int, fid: int) -> Tuple[ioh.problem.PBO, ioh.logger.Analyzer]:
    # Declaration of problems to be tested.
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.PBO)

    # Create default logger compatible with IOHanalyzer
    # `root` indicates where the output files are stored.
    # `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload it to IOHanalyzer.
    l = logger.Analyzer(
        root="data",  # the working directory in which a folder named `folder_name` (the next argument) will be created to store data
        folder_name="run",  # the folder name to which the raw performance data will be stored
        algorithm_name="genetic_algorithm",  # name of your algorithm
        algorithm_info="Practical assignment of the EA course",
    )
    # attach the logger to the problem
    problem.attach_logger(l)
    return problem, l


if __name__ == "__main__":
    # this how you run your algorithm with 20 repetitions/independent run
    # create the LABS problem and the data logger
    F18, _logger = create_problem(dimension=50, fid=18)
    for run in range(20): 
        studentnumber1_studentnumber2_GA(F18)
        F18.reset() # it is necessary to reset the problem after each independent run
    _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder

    # create the N-Queens problem and the data logger
    F23, _logger = create_problem(dimension=49, fid=23)
    for run in range(20): 
        studentnumber1_studentnumber2_GA(F23)
        F23.reset()
    _logger.close()