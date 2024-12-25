from typing import List, Tuple
import numpy as np
from ioh import get_problem, logger, ProblemClass
from s3997545_s4473590_GA import s3997545_s4473590_GA, create_problem

budget = 1000000

hyperparameter_space = {
    "population_size": [50, 100,200],
    "mutation_rate": [ 0.3,0.9],
    "crossover_rate": [0.3, 0.5, 0.9]
}

# Evaluate hyperparameters across both problems
def evaluate_hyperparameters(problem_ids: List[int], dimensions: List[int]) -> Tuple[int, float, float]:
    best_score = float('-inf')
    best_params = None

    for pop_size in hyperparameter_space['population_size']:
        for mutation_rate in hyperparameter_space['mutation_rate']:
            for crossover_rate in hyperparameter_space['crossover_rate']:
                # Accumulate scores across all problems and independent runs
                total_score = 0

                for problem_id, dimension in zip(problem_ids, dimensions):
                    # Create problem and logger for each problem
                    problem, _logger = create_problem(dimension=dimension, fid=problem_id)

                    for _ in range(5):  # Perform 5 independent runs
                        s3997545_s4473590_GA(problem, pop_size, mutation_rate, crossover_rate)
                        total_score += problem.state.current_best.y
                        problem.reset()  # Reset the problem for the next run

                    # Close logger after each problem
                    _logger.close()

                avg_score = total_score / (5 * len(problem_ids)) 

                # Update best parameters
                if avg_score > best_score:
                    best_score = avg_score
                    best_params = (pop_size, mutation_rate, crossover_rate)

                print(f"Evaluated params: pop_size={pop_size}, mutation_rate={mutation_rate}, "
                      f"crossover_rate={crossover_rate}, avg_score={avg_score}")

    return best_params

if __name__ == "__main__":
    # Define problems and dimensions
    problem_ids = [18, 23]  # F18 (LABS), F23 (N-Queens)
    dimensions = [50, 49]  # Dimensions for each problem

    print("Tuning hyperparameters for both F18 (LABS) and F23 (N-Queens) problems")
    population_size, mutation_rate, crossover_rate = evaluate_hyperparameters(problem_ids, dimensions)
    print("Best hyperparameters for both problems:")
    print("Population Size:", population_size)
    print("Mutation Rate:", mutation_rate)
    print("Crossover Rate:", crossover_rate)
