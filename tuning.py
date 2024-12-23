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

# Hyperparameter tuning function
def tune_hyperparameters() -> Tuple[int, float, float]:
    best_score = float('inf')
    best_params = None

    # Create LABS problem (F18) and N-Queens problem (F23)
    F18, _logger_f18 = create_problem(dimension=50, fid=18)
    F23, _logger_f23 = create_problem(dimension=49, fid=23)

    for pop_size in hyperparameter_space['population_size']:
        for mutation_rate in hyperparameter_space['mutation_rate']:
            for crossover_rate in hyperparameter_space['crossover_rate']:
                # Accumulate scores across both problems
                total_score = 0

                # Run on LABS problem (F18)
                for _ in range(5):  # Perform 5 independent runs
                    studentnumber1_studentnumber2_GA(F18, pop_size, mutation_rate, crossover_rate)
                    total_score += F18.state.current_best.y
                    F18.reset()  # Reset the problem for the next run

                # Run on N-Queens problem (F23)
                for _ in range(5):  # Perform 5 independent runs
                    studentnumber1_studentnumber2_GA(F23, pop_size, mutation_rate, crossover_rate)
                    total_score += F23.state.current_best.y
                    F23.reset()  # Reset the problem for the next run

                # Calculate average score
                avg_score = total_score / 10  # 10 total runs (5 for each problem)

                # Update best parameters
                if avg_score < best_score:
                    best_score = avg_score
                    best_params = (pop_size, mutation_rate, crossover_rate)

                print(f"Evaluated params: pop_size={pop_size}, mutation_rate={mutation_rate}, "
                      f"crossover_rate={crossover_rate}, avg_score={avg_score}")

    # Close loggers
    _logger_f18.close()
    _logger_f23.close()

    return best_params


if __name__ == "__main__":
    # Tune hyperparameters
    population_size, mutation_rate, crossover_rate = tune_hyperparameters()
    print("Best hyperparameters:")
    print("Population Size:", population_size)
    print("Mutation Rate:", mutation_rate)
    print("Crossover Rate:", crossover_rate)
