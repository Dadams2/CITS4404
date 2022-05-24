from re import U
import numpy as np
import random

from Camo_Worm import random_worm, Camo_Worm 
from Drawing import Drawing, prep_image
from cost_function import costfn

class Algorithm():
    clew: list[Camo_Worm] = []
    image: np.ndarray = np.zeros((1,1))
    init_params: tuple = (1, 30, 3)

    INTERNAL_PARAMS = ['x', 'y', 'colour']
    OTHER_PARAMS = ['r', 'theta', 'width', 'dr', 'dgamma']
    ALL_PARAMS = ['x', 'y', 'colour', 'r', 'theta', 'width', 'dr', 'dgamma']
    POINTS_ON_CURVE_COUNT = 100

    # initialise a clew of random worms
    def __init__(self, size, image, init_params):
        self.init_params = init_params
        self.image = image

        for i in range(size):
            worm = random_worm(self.image.shape, self.init_params)
            self.clew.append(worm)
        self.image = image

    # fitness function
    def fitness(self, clew):
        # loop over clew
        cost_score = 0
        for index, item in enumerate(clew):
            # calculate fitness
            cost_score += costfn(self.clew, index, self.image.shape, self.image)

        return cost_score

    #crossover function
    def crossover(self, parent1, parent2, probability=0.5):
        child1_params = {}
        child2_params = {}

        parent_pair = [parent1, parent2]
        
        if random.random() < probability:
            # choosing a crossover point
            # choose a random integer between 0 and the length of params
            param_chosen = self.ALL_PARAMS[random.randrange(1, len(self.ALL_PARAMS) - 1)]

            # clone parent params to child
            for param in self.ALL_PARAMS:
                child1_params[param] = parent_pair[0].__getattribute__(param)
                child2_params[param] = parent_pair[1].__getattribute__(param)

            # loop until param and swap param values between parents
            for index, param in enumerate(self.ALL_PARAMS):
                # swap parameter values
                child1_params[param] = parent_pair[1].__getattribute__(param)
                child2_params[param] = parent_pair[0].__getattribute__(param)

                if param == param_chosen:
                    break

            # create params from child params
            child1 = Camo_Worm(child1_params['x'], child1_params['y'], child1_params['r'], child1_params['theta'], child1_params['dr'], child1_params['dgamma'], child1_params['width'], child1_params['colour'])
            child2 = Camo_Worm(child2_params['x'], child2_params['y'], child2_params['r'], child2_params['theta'], child2_params['dr'], child2_params['dgamma'], child2_params['width'], child2_params['colour'])

            return child1, child2

        return parent1, parent2

    # mutate function
    # add a probability
    def mutate(self, member, probability=0.5):
        if random.random() < probability:
            random_param = random.choice(self.ALL_PARAMS)
            rand_worm = random_worm(self.image.shape, self.init_params)
            member.__setattr__(random_param, rand_worm.__getattribute__(random_param))

        return member

    # create_new_population function
    def create_next_generation(self, population):
        # choose top 50 percent of population
        best_population = population[:int(len(population) / 2)]

        # step 1 - an array of new population
        new_population = []
        # step 2 - loop over pairs of population
        for i in range(0, len(best_population), 2):
            # step 3 - crossover
            child1, child2 = self.crossover(best_population[i], best_population[i+1], probability=1)

            # step 4 - mutate
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)

            # step 5 - add to new population
            new_population.append(child1)
            new_population.append(child2)

        # add 50% random worms to new population
        for i in range(int(len(population) / 2)):
            new_population.append(random_worm(self.image.shape, self.init_params))

        return new_population


    def internal_cost(self, population, index):
        worm = population[index]
        cost = 0
        # worm should be longer 
        length = worm.r * 2 
        width = worm.width 
        
        return (length * width) / (self.image.shape[0])

    def environmental_cost(self, population, index):
        # get intermediate points
        worm = population[index]
        points = worm.intermediate_points(self.POINTS_ON_CURVE_COUNT)

        # loop over intermediate points
        # find intensity
        diff = 0
        prev = 0
        for point in points:
            x = int(max(0, min(image.shape[1]-1, point[0])))
            y = int(max(0, min(image.shape[0]-1, point[1])))
            
            diff += abs(int(image[y, x]) - int(prev))
            prev = image[y, x]
        
        # smootheness score - not sure this is a good way to scale it to be 0 ... 1
        return 1 - 1/diff
        
    def costfn(self, population, index):
        # longer the length is, higher the score to return
        return self.internal_cost(population, index)

    def run_algo(self, number_of_iterations, population_size):
        # Step 1 - Generate a random population
        population = []
        for i in range(population_size):
            population.append(random_worm(self.image.shape, self.init_params))
        
        # Step 2 - Evaluate fitness
        fitness = [self.costfn(population, i ) for i in range(len(population))]

        for i in range(number_of_iterations):
            # Step 4 - Get Fitness
            fitness = [self.costfn(population, i) for i in range(len(population))]

            # Step 3 - Sort by the fitness
            zipped = zip(population, fitness)
            zipped = sorted(zipped, key=lambda x: x[1], reverse=True)
            population, fitness = zip(*zipped)

            # Step 3 - Create next generation
            population = self.create_next_generation(population)

        self.clew = population

image = prep_image()
img_shape = image.shape

algo = Algorithm(20, image, (1, 30, 3)) 
clew = algo.clew

# display clew on image
drawing = Drawing(image)
drawing.add_worms(clew)
drawing.show()

algo.run_algo(100, 20)

final_clew = algo.clew

drawing = Drawing(image)
drawing.add_worms(final_clew)
drawing.show()


print("### Length of initial clew ###")
print(len(clew))

print("### Length of Final Clew ###")
print(len(final_clew))