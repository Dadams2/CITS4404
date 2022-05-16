import numpy as np
import random

from Camo_Worm import *
from Drawing import Drawing
from image import prep_image
# from cost_function_1 import costfn
from cost_function_2 import costfn


image = prep_image()
img_shape = image.shape


def draw_worms(this_clew, i):
    drawing = Drawing(image)
    drawing.add_worms(this_clew)
    drawing.show(save=f'src/img_results/output_{i}.png')


# Cost Functions
def evaluation(this_clew):
    return [costfn(this_clew, i, img_shape, image, w_env=10) for i, _worm in enumerate(this_clew)]


def selection(this_clew, costs):
    
    costs = evaluation(this_clew)
    zipped_clew = zip(this_clew, costs)
    sorted_clew = sorted(zipped_clew, key=lambda zipped: zipped[1])

    return [worm for worm, _cost in sorted_clew[:int(len(this_clew)*0.50)]]  # 20% of the best worms in a clew


# Crossover and Mutation
def next_clew(best_clew, imshape, num_child=100):
    res = []
    for _ in range(num_child):
        # new_worm = random_worm(imshape, (40, 30, 1))
        new_params = {}
        # selects 2 parents for crossover
        parents = random.sample(best_clew, k=2)

        for param in ["x", "y", "r", "theta", "dr", "dgamma", "width", "colour"]:
        # for attr in [['x', 'y', 'colour'], ['r', 'theta', 'width'], ['dr', 'dgamma']]:
            
            good_worm = random.choice(parents)
            # for param in attr:
            # Mutation
            if random.random() < 0.2:
                # TODO: Make mutation closer to good worm's attr (in most cases +/- 10%)
                new_param = good_worm.__getattribute__(param) * ((random.random() * 0.40) + 0.80)   # mutate between 90 - 110% of good worms attr value 
            else:
                new_param = good_worm.__getattribute__(param)
            # new_param = good_worm.__getattribute__(param)

            new_params[param] = new_param

        # print(new_params)
        
        # clamp worm values so they dont become invalid for problem space
        new_worm = Camo_Worm(
            x= min(new_params["x"], imshape[1]),
            y= min(new_params["y"], imshape[0]), 
            r= new_params["r"], 
            theta= new_params["theta"], 
            deviation_r= new_params["dr"], 
            deviation_gamma= min(new_params["dgamma"], np.pi), 
            width= new_params["width"],
            colour= min(new_params["colour"], 1.0)
            )
        res.append(new_worm)
    return res  



def next_clew2(best_clew, imshape, num_child=100):
    res = []
    for _ in range(int(num_child/2)):
        # new_worm = random_worm(imshape, (40, 30, 1))
        child1_params = {}
        child2_params = {}
        # selects 2 parents for crossover
        parents = random.sample(best_clew, k=2)

        for param in ["x", "y", "r", "theta", "dr", "dgamma", "width", "colour"]:
        # for attr in [['x', 'y', 'colour'], ['r', 'theta', 'width'], ['dr', 'dgamma']]:
            
            selection = int(random.random())
            parent1 = parents[selection]
            parent2 = parents[1-selection]
            
            child1_params[param] = parent1.__getattribute__(param)
            child2_params[param] = parent2.__getattribute__(param)
            # for param in attr:
        

            # Mutation
            if random.random() < 1/8:
                child1_params[param] = child1_params[param] * ((random.random() / 5) + 0.90)  # mutate between 90 - 110% of good worms attr value 

            if random.random() < 1/8:
                child2_params[param] = child2_params[param] * ((random.random() / 5) + 0.90)  # mutate between 90 - 110% of good worms attr value 

        # print(new_params)
        
        # clamp worm values so they dont become invalid for problem space
        new_worm1 = Camo_Worm(
            x= min(child1_params["x"], imshape[1]),
            y= min(child1_params["y"], imshape[0]), 
            r= child1_params["r"], 
            theta= child1_params["theta"], 
            deviation_r= child1_params["dr"], 
            deviation_gamma= min(child1_params["dgamma"], np.pi), 
            width= child1_params["width"],
            colour= min(child1_params["colour"], 1.0)
            )
        res.append(new_worm1)

        new_worm2 = Camo_Worm(
            x= min(child2_params["x"], imshape[1]),
            y= min(child2_params["y"], imshape[0]), 
            r= child2_params["r"], 
            theta= child2_params["theta"], 
            deviation_r= child2_params["dr"], 
            deviation_gamma= min(child2_params["dgamma"], np.pi), 
            width= child2_params["width"],
            colour= min(child2_params["colour"], 1.0)
            )
        res.append(new_worm2)
        
    return res[:num_child]



def evolutionary_algorithm(iterations: int):

    this_clew = initialise_random_clew(100, image.shape, (40, 30, 1))
    
    for i in range(iterations):
        draw_worms(this_clew, i)

        _costs = evaluation(this_clew)
        _bests = selection(this_clew, _costs)
        
        this_clew = next_clew(_bests, img_shape, num_child=100)



evolutionary_algorithm(10)




































# # Random Clew
# clew = initialise_random_clew(100, image.shape, (40, 30, 1))
# drawing = Drawing(image)
# drawing.add_worms(clew)
# drawing.show()

# # Worm Example
# worm1 = Camo_Worm(200, 100, 50, np.pi/6, 70, np.pi/3, 10, 0.8)
# drawing1 = Drawing(image)
# drawing1.add_worm_with_details(worm1)

# # Another Worm Example
# worm2 = Camo_Worm(350, 100, 300, 0, 70, -np.pi/2, 5, 0.8)
# drawing2 = Drawing(image)
# drawing2.add_worm_with_details(worm2)

# costfn(clew)