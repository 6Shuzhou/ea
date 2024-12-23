from typing import List, Tuple
import numpy as np
from ioh import get_problem, logger, ProblemClass
from GA import studentnumber1_studentnumber2_GA, create_problem

budget = 1000000

# Hyperparameters to tune
hyperparameter_space = {
    "population_size": [50, 100, 200],
    "mutation_rate": [0.3, 0.5, 0.7],
    "crossover_rate": [0.5, 0.7, 0.9]
}

# Hyperparameter tuning function for a single problem
def tune_hyperparameters_for_problem(problem_id: int, dimension: int) -> Tuple[int, float, float]:
    best_score = float('-inf')
    best_params = None

    # Create problem and logger
    problem, _logger = create_problem(dimension=dimension, fid=problem_id)

    for pop_size in hyperparameter_space['population_size']:
        for mutation_rate in hyperparameter_space['mutation_rate']:
            for crossover_rate in hyperparameter_space['crossover_rate']:
                # Accumulate scores across independent runs
                total_score = 0

                for _ in range(5):  # Perform 5 independent runs
                    studentnumber1_studentnumber2_GA(problem, pop_size, mutation_rate, crossover_rate)
                    total_score += problem.state.current_best.y
                    problem.reset()  # Reset the problem for the next run

                # Calculate average score
                avg_score = total_score / 5  # 5 total runs

                # Update best parameters
                if avg_score > best_score:
                    best_score = avg_score
                    best_params = (pop_size, mutation_rate, crossover_rate)

                print(f"Problem {problem_id}, evaluated params: pop_size={pop_size}, mutation_rate={mutation_rate}, "
                      f"crossover_rate={crossover_rate}, avg_score={avg_score}")

    # Close logger
    _logger.close()

    return best_params

if __name__ == "__main__":
    # Tune hyperparameters for F18 (LABS)
    print("Tuning hyperparameters for F18 (LABS problem)")
    population_size_f18, mutation_rate_f18, crossover_rate_f18 = tune_hyperparameters_for_problem(problem_id=18, dimension=50)
    print("Best hyperparameters for F18:")
    print("Population Size:", population_size_f18)
    print("Mutation Rate:", mutation_rate_f18)
    print("Crossover Rate:", crossover_rate_f18)

    # Tune hyperparameters for F23 (N-Queens)
    print("Tuning hyperparameters for F23 (N-Queens problem)")
    population_size_f23, mutation_rate_f23, crossover_rate_f23 = tune_hyperparameters_for_problem(problem_id=23, dimension=49)
    print("Best hyperparameters for F23:")
    print("Population Size:", population_size_f23)
    print("Mutation Rate:", mutation_rate_f23)
    print("Crossover Rate:", crossover_rate_f23)
