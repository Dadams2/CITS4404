import numpy as np
import random
from datetime import datetime
from tqdm import tqdm

from Camo_Worm import *
from Drawing import Drawing, prep_image, get_smoothed
from cost_function import costfn


image = prep_image()
smoothed_image = get_smoothed(image)
img_shape = image.shape
smoothed_shape = smoothed_image.shape


def draw_worms(this_clew, title, clew_size=None, iterations=None, show="yes", save="no"):
    drawing = Drawing(image)
    if clew_size is None or iterations is None:
        drawing.add_title(f"{title}")
    else:
        drawing.add_title(f"{title} - Clew Size: {clew_size} - Iterations: {iterations}")
    drawing.add_worms(this_clew)
    if save == "yes": 
        date_string = datetime.now().strftime("%m%d%Y-%H%M") # current date and time
        drawing.save(f'src/img_results/output_{date_string}_{title}_.png')
    if show == "yes": drawing.show()



def mutation(param, imshape, init_params) -> float:
    """
        Randomly mutates a parameter based on the limits of that parameter
        Parameters:
            param (string): will be one of the following values --> 'x', 'y', 'colour', 'r', 'theta', 'width', 'dr', 'dgamma'
    """
    rng = np.random.default_rng() 
    (radius_std, deviation_std, width_theta) = init_params
    if param == 'x':
        return rng.random() * imshape[1]
    elif param == 'y':
        return rng.random() * imshape[0]
    elif param == 'colour':
        return rng.random()
    elif param == 'r':
        return radius_std * np.abs(rng.standard_normal())
    elif param == 'theta':
        return rng.random() * np.pi
    elif param == 'width':
        return width_theta * rng.standard_gamma(3)
    elif param == 'dr':
        return deviation_std * np.abs(rng.standard_normal())
    elif param == "dgamma":
        return rng.random() * np.pi


def new_child_clew(best_clew: list[Camo_Worm], init_params):

    # CONSTANTS
    internal_PARAMS = ['x', 'y', 'colour']
    other_PARAMS = ['r', 'theta', 'width', 'dr', 'dgamma']
    all_PARAMS = ['x', 'y', 'colour', 'r', 'theta', 'width', 'dr', 'dgamma']
    
    index = 0;
    clew_size = len(best_clew)

    new_children = []

    for index in range(0, clew_size, 2):
        
        if (index+1) >= clew_size:  # To take care of the out of bound problem in case of an odd number 
            break

        ###############################################
        # Crossover

        child1_params = {}
        child2_params = {}

        parent_pair = random.sample(best_clew, k=2)  # Completely Random Parents
        # parent_pair = [best_clew[index], best_clew[index+1]]  # Iterating through all worms in best_clew


        selection = int(random.random())
        for param in internal_PARAMS:
            child1_params[param] = parent_pair[selection].__getattribute__(param) * (random.random() * 1) + 0.5
            child2_params[param] = parent_pair[1-selection].__getattribute__(param) * (random.random() * 1) + 0.5

        
        for param in other_PARAMS:
            selection = int(random.random())
            child1_params[param] = parent_pair[selection].__getattribute__(param)
            child2_params[param] = parent_pair[1-selection].__getattribute__(param)

            if random.random() < 0.01:
               child1_params[param] = mutation(param, img_shape, init_params)

            if random.random() < 0.01:
               child2_params[param] = mutation(param, img_shape, init_params)


        # ###############################################
        # # Mutation - (Hadi): just trying to get more mutations happening

        # for param in all_PARAMS:
        #     if random.random() < 0.01:
        #         # child1_params[param] *= ((random.random() * 0.6) + 0.7)  # mutate between 90 - 110% of good worms attr value 
        #        child1_params[param] = mutation(param, img_shape, init_params) # child1_params[param] * ((random.random() / 5) + 0.90)  # mutate between 90 - 110% of good worms attr value 

        #     if random.random() < 0.01:
        #         # child2_params[param] *= ((random.random() * 0.6) + 0.7)  # mutate between 90 - 110% of good worms attr value 
        #         child2_params[param] = mutation(param, img_shape, init_params) # child2_params[param] * ((random.random() / 5) + 0.90)  # mutate between 90 - 110% of good worms attr value 



        ###############################################
        # Making Children

        new_worm1 = Camo_Worm(
                x= min(child1_params["x"], img_shape[1]),
                y= min(child1_params["y"], img_shape[0]), 
                r= child1_params["r"], 
                theta= child1_params["theta"], 
                deviation_r= child1_params["dr"], 
                deviation_gamma= min(child1_params["dgamma"], np.pi), 
                width= child1_params["width"],
                colour= min(child1_params["colour"], 1.0)
                )
        new_children.append(new_worm1)

        new_worm2 = Camo_Worm(
            x= min(child2_params["x"], img_shape[1]),
            y= min(child2_params["y"], img_shape[0]), 
            r= child2_params["r"], 
            theta= child2_params["theta"], 
            deviation_r= child2_params["dr"], 
            deviation_gamma= min(child2_params["dgamma"], np.pi), 
            width= child2_params["width"],
            colour= min(child2_params["colour"], 1.0)
            )
        new_children.append(new_worm2)

    return new_children


def evolutionary_algorithm(iterations: int, clew_size: int):

    selection_VALUE = 0.30          # Constant - will select selection_VALUE - elite of the best worms in a clew
    elite = 0.1                     # keep top 10% of each generation w/o modification (elitism concept)
    init_params = (40, 30, 1)

    this_clew : list[Camo_Worm] = initialise_random_clew(clew_size, image.shape, init_params=init_params)
    draw_worms(this_clew, title="initial")


    # for i in tqdm(range(iterations)):
    for i in range(iterations):

        

        ###############################################
        # Evaluation and Cost Functions

        costs = [costfn(this_clew, i, img_shape, image, smoothed_image, smoothed_shape) for i, _worm in enumerate(this_clew)]
        print(sum(costs))


        ###############################################
        # Selection (picking best parents/worms)

        zipped_clew = zip(this_clew, costs)
        sorted_clew = sorted(zipped_clew, key=lambda zipped: zipped[1])

        elite_number = int(len(this_clew) * elite)
        best_clew = [worm for worm, _cost in sorted_clew[elite_number:int(len(this_clew) * selection_VALUE)]]
        # remove parents from the clew


        ###############################################
        # Crossover + Mutation

        child_clew = new_child_clew(best_clew, init_params)

        for index, child in enumerate(child_clew):
            sorted_clew.append((child, costfn(child_clew, index, img_shape, image, smoothed_image, smoothed_shape)))
        sorted_clew = sorted(sorted_clew, key=lambda x: x[1])

        this_clew = [worm for worm, _cost in sorted_clew[:clew_size]]


        ###############################################
        # Printing/showing/saving images for all iteration

        # draw_worms(this_clew, title=i, show="no", save="yes")

    costs = [costfn(this_clew, i, img_shape, image, smoothed_image, smoothed_shape) for i, _worm in enumerate(this_clew)]
    print(sum(costs))
    draw_worms(this_clew, title=f"final-{iterations}iter-{clew_size}clew-", save="yes")

    for i, worm in enumerate(this_clew):
        print(i, ":::", worm.centre_point())

evolutionary_algorithm(iterations=200, clew_size=200)
