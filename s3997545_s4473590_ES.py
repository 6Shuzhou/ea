import numpy as np
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
from ioh import get_problem, logger, ProblemClass

budget = 50000
dimension = 10
np.random.seed(42)

def s3997545_s4473590_ES(problem):
    mu = 50  # Population size (parents)
    lambda_ = 50  # Offspring size
    sigma = 0.1  # Mutation strength (standard deviation)
    
    population = np.random.uniform(
        low=-5, high=5, size=(mu, problem.meta_data.n_variables)
    )
    
    # Evaluate initial population
    fitness = np.array([problem(ind) for ind in population])
    
    while problem.state.evaluations +lambda_ <= budget:
        # Parent selection (randomly choose parents for recombination)
        parents = population[np.random.choice(mu, size=lambda_, replace=True)]
        
        # Recombination (arithmetic crossover)
        offspring = (parents + parents[np.random.permutation(lambda_)]) / 2
        
        # Mutation (Gaussian perturbation)
        offspring += np.random.normal(0, sigma, offspring.shape)
        
        # Ensure offspring are within bounds
        offspring = np.clip(offspring, -5, 5)
        
        # Evaluate offspring
        offspring_fitness = np.array([problem(ind) for ind in offspring])
        
        # Combine parents and offspring (mu + lambda selection)
        combined_population = np.vstack([population, offspring])
        combined_fitness = np.hstack([fitness, offspring_fitness])
        
        # Select the best mu individuals for the next generation
        best_indices = np.argsort(combined_fitness)[:mu]
        population = combined_population[best_indices]
        fitness = combined_fitness[best_indices]



def create_problem(fid: int):
    # Declaration of problems to be tested.
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.BBOB)

    # Create default logger compatible with IOHanalyzer
    # `root` indicates where the output files are stored.
    # `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload it to IOHanalyzer.
    l = logger.Analyzer(
        root="data",  # the working directory in which a folder named `folder_name` (the next argument) will be created to store data
        folder_name="run",  # the folder name to which the raw performance data will be stored
        algorithm_name="evolution strategy",  # name of your algorithm
        algorithm_info="Practical assignment part2 of the EA course",
    )
    # attach the logger to the problem
    problem.attach_logger(l)
    return problem, l


if __name__ == "__main__":
    # this how you run your algorithm with 20 repetitions/independent run
    F23, _logger = create_problem(23)
    for run in range(20): 
        s3997545_s4473590_ES(F23)
        F23.reset() # it is necessary to reset the problem after each independent run
    _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder


