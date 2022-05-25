import numpy as np
import random

from Camo_Worm import *
from Drawing import Drawing, prep_image
from test_cost_function import internal_cost_score, group_cost_score, environmental_cost_score


image = prep_image()
img_shape = image.shape


########################################
# Random Clew
# clew = initialise_random_clew(100, image.shape, (40, 30, 1))
# drawing = Drawing(image)
# drawing.add_worms(clew)
# drawing.show()


worm_internal = Camo_Worm(350, 100, 50, 0, -20, -np.pi/2, 5, 0.8)
print("worm_internal        :", internal_cost_score(worm_internal, img_shape))
drawing3 = Drawing(image)
drawing3.add_worm_with_details(worm_internal)


worm_internal_1 = Camo_Worm(350, 100, 200, 0, -20, -np.pi/2, 5, 0.8)
print("worm_internal_1      :", internal_cost_score(worm_internal_1, img_shape))
drawing3 = Drawing(image)
drawing3.add_worm_with_details(worm_internal_1)


worm_internal_2 = Camo_Worm(350, 100, 50, np.pi/2, -20, -np.pi/2, 5, 0.8)
print("worm_internal_2      :", internal_cost_score(worm_internal_2, img_shape))
drawing3 = Drawing(image)
drawing3.add_worm_with_details(worm_internal_2)


worm_internal_3 = Camo_Worm(350, 100, 50, 0, -100, -np.pi/2, 5, 0.8)
print("worm_internal_3      :", internal_cost_score(worm_internal_3, img_shape))
drawing3 = Drawing(image)
drawing3.add_worm_with_details(worm_internal_3)


worm_internal_4 = Camo_Worm(350, 100, 50, np.pi/2, -100, -np.pi/2, 5, 0.8)
print("worm_internal_4      :", internal_cost_score(worm_internal_4, img_shape))
drawing3 = Drawing(image)
drawing3.add_worm_with_details(worm_internal_4)







worm_env = Camo_Worm(350, 45, 150, 0, 10, -np.pi/2, 5, 0.8)
print("worm_env       :", environmental_cost_score(worm_env, image, img_shape))
drawing3 = Drawing(image)
drawing3.add_worm_with_details(worm_env)



worm_env_2 = Camo_Worm(350, 230, 150, 0, 10, -np.pi/2, 5, 0.8)
print("worm_env_2     :", environmental_cost_score(worm_env_2, image, img_shape))
drawing3 = Drawing(image)
drawing3.add_worm_with_details(worm_env_2)


worm_env_3 = Camo_Worm(350, 230, 150, 0, 10, -np.pi/2, 5, 0.1)
print("worm_env_3     :", environmental_cost_score(worm_env_3, image, img_shape))
drawing3 = Drawing(image)
drawing3.add_worm_with_details(worm_env_3)


worm_env_3 = Camo_Worm(350, 180, 50, 0, 50, -np.pi/2, 5, 0.1)
print("worm_env_3     :", environmental_cost_score(worm_env_3, image, img_shape))
drawing3 = Drawing(image)
drawing3.add_worm_with_details(worm_env_3)




w = 60
h = 30


x = 350
y = 120
diff_x = 100
diff_y = 50

worm_group = Camo_Worm(x, y, 50, 0, 10, -np.pi/2, 5, 0.8)
clew = [
    Camo_Worm(x + diff_x, y + diff_y, 20, 0, 10, -np.pi/2, 5, 0.8),
    Camo_Worm(x - diff_x, y + diff_y, 20, 0, 10, -np.pi/2, 5, 0.8),
    Camo_Worm(x + diff_x, y - diff_y, 20, 0, 10, -np.pi/2, 5, 0.8),
    Camo_Worm(x - diff_x, y - diff_y, 20, 0, 10, -np.pi/2, 5, 0.8)
]
print("worm_group       :", group_cost_score(clew, worm_group))
drawing = Drawing(image)
drawing.add_worms(clew)
drawing.add_rect(x + diff_x - w/2, y + diff_y - h/2, w, h)
drawing.add_rect(x - diff_x - w/2, y + diff_y - h/2, w, h)
drawing.add_rect(x + diff_x - w/2, y - diff_y - h/2, w, h)
drawing.add_rect(x - diff_x - w/2, y - diff_y - h/2, w, h)
drawing.add_worm_with_details(worm_group)


x = 350
y = 120
diff_x = 200
diff_y = 100

worm_group_2 = Camo_Worm(350, 120, 50, 0, 10, -np.pi/2, 5, 0.8)
clew_2 = [
    Camo_Worm(x + diff_x, y + diff_y, 20, 0, 10, -np.pi/2, 5, 0.8),
    Camo_Worm(x - diff_x, y + diff_y, 20, 0, 10, -np.pi/2, 5, 0.8),
    Camo_Worm(x + diff_x, y - diff_y, 20, 0, 10, -np.pi/2, 5, 0.8),
    Camo_Worm(x - diff_x, y - diff_y, 20, 0, 10, -np.pi/2, 5, 0.8)
]
print("worm_group_2     :", group_cost_score(clew_2, worm_group_2))
drawing2 = Drawing(image)
drawing2.add_worms(clew_2)
drawing2.add_rect(x + diff_x - w/2, y + diff_y - h/2, w, h)
drawing2.add_rect(x - diff_x - w/2, y + diff_y - h/2, w, h)
drawing2.add_rect(x + diff_x - w/2, y - diff_y - h/2, w, h)
drawing2.add_rect(x - diff_x - w/2, y - diff_y - h/2, w, h)
drawing2.add_worm_with_details(worm_group_2)

