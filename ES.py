import numpy as np
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
from ioh import get_problem, logger, ProblemClass

budget = 50000
dimension = 10
np.random.seed(42)

def studentnumber1_studentnumber2_ES(problem):
    population_size = 50
    sigma = 0.2  # 控制变异强度
    mutation_rate = 0.5  # 位翻转概率

    # Step 1: 初始化种群
    population = np.random.rand(population_size, dimension)
    
    while problem.state.evaluations < budget:
        # Step 2: 变异操作（连续）
        mutated_population = population + np.random.normal(0, sigma, size=population.shape)
        mutated_population = np.clip(mutated_population, 0, 1)

        # Step 3: 将连续解转换为布尔解
        bit_population = (population > 0.5).astype(int)
        mutated_bit_population = (mutated_population > 0.5).astype(int)

        # Step 4: 对布尔解进行位翻转变异
        for i in range(population_size):
            if np.random.rand() < mutation_rate:
                mutation_point = np.random.randint(dimension)
                mutated_bit_population[i][mutation_point] ^= 1  # 翻转位

        # Step 5: 评价适应度
        combined_population = np.vstack((bit_population, mutated_bit_population))
        fitness = np.array([problem(ind) for ind in combined_population])

        # Step 6: \( (\mu + \lambda) \) 选择
        best_indices = np.argsort(fitness)[-population_size:]  # 选择适应度最高的个体
        population = combined_population[best_indices]



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
        studentnumber1_studentnumber2_ES(F23)
        F23.reset() # it is necessary to reset the problem after each independent run
    _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder


