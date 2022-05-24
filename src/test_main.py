import numpy as np
import random

from Camo_Worm import *
from Drawing import Drawing, prep_image
from test_cost_function import internal_cost_score


image = prep_image()
img_shape = image.shape


########################################
# Random Clew
# clew = initialise_random_clew(100, image.shape, (40, 30, 1))
# drawing = Drawing(image)
# drawing.add_worms(clew)
# drawing.show()


worm_internal = Camo_Worm(350, 100, 50, 0, -20, -np.pi/2, 5, 0.8)
drawing3 = Drawing(image)
drawing3.add_worm_with_details(worm_internal)

print("worm_internal: \t\t", internal_cost_score(worm_internal, img_shape))


worm_internal_1 = Camo_Worm(350, 100, 200, 0, -20, -np.pi/2, 5, 0.8)
drawing3 = Drawing(image)
drawing3.add_worm_with_details(worm_internal_1)

print("worm_internal_1: \t", internal_cost_score(worm_internal_1, img_shape))


worm_internal_2 = Camo_Worm(350, 100, 50, np.pi/2, -20, -np.pi/2, 5, 0.8)
drawing3 = Drawing(image)
drawing3.add_worm_with_details(worm_internal_2)

print("worm_internal_2: \t", internal_cost_score(worm_internal_2, img_shape))


worm_internal_3 = Camo_Worm(350, 100, 50, 0, -100, -np.pi/2, 5, 0.8)
drawing3 = Drawing(image)
drawing3.add_worm_with_details(worm_internal_3)

print("worm_internal_3: \t", internal_cost_score(worm_internal_3, img_shape))
