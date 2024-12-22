from typing import Tuple 
import numpy as np

import ioh
from ioh import get_problem, logger, ProblemClass

budget = 5000

np.random.seed(42)

def studentnumber1_studentnumber2_GA(
    problem: ioh.problem.PBO,
    population_size: int = 50,
    mutation_rate: float = 0.2,
    crossover_rate: float = 0.5,
    budget: int = 5000
) -> None:
    # Step 1: Initialize population
    population = np.random.randint(2, size=(population_size, problem.meta_data.n_variables))
    
    # Step 2: Evaluate initial population
    fitness = np.array([problem(ind) for ind in population])
    
    while problem.state.evaluations < budget:
        # Step 3: Selection (Roulette Wheel Selection)
        total_fitness = fitness.sum()
        if total_fitness <= 0 or np.any(fitness < 0):
            # Handle edge case where all fitnesses are zero
            probabilities = np.ones(population_size) / population_size
        else:
            probabilities = fitness / total_fitness  # Normalize fitness to create selection probabilities
        
        selected_indices = np.random.choice(population_size, size=population_size, replace=True, p=probabilities)
        parents = population[selected_indices]

        # # Step 4: Crossover
        # offspring = []
        # for i in range(0, population_size, 2):
        #     if np.random.rand() < crossover_rate and i + 1 < population_size:
        #         point = np.random.randint(1, problem.meta_data.n_variables)
        #         parent1, parent2 = parents[i], parents[i + 1]
        #         child1 = np.concatenate((parent1[:point], parent2[point:]))
        #         child2 = np.concatenate((parent2[:point], parent1[point:]))
        #         offspring.extend([child1, child2])
        #     else:
        #         offspring.extend([parents[i], parents[i + 1]])
        
        # offspring = np.array(offspring[:population_size])  # Ensure size consistency
        
        # Step 4: Crossover (Uniform Crossover)
        offspring = []
        for i in range(0, population_size, 2):
            if i + 1 < population_size:  # 确保父代数量为偶数
                parent1, parent2 = parents[i], parents[i + 1]
                if np.random.rand() < crossover_rate:
            # 生成随机掩码
                    mask = np.random.randint(2, size=problem.meta_data.n_variables)
            # 根据掩码生成子代
                    child1 = np.where(mask == 1, parent1, parent2)
                    child2 = np.where(mask == 1, parent2, parent1)
                else:
            # 不交叉则直接保留父代
                   child1, child2 = parent1, parent2
                offspring.extend([child1, child2])
            else:
        # 若剩余父代数量为奇数，保留最后一个父代
                offspring.append(parents[i])

        offspring = np.array(offspring[:population_size])  # 确保子代数量一致

        
        
        
        
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
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.PBO)
    l = logger.Analyzer(
        root="data",  
        folder_name="run", 
        algorithm_name="genetic_algorithm",  
        algorithm_info="Practical assignment of the EA course",
    )
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