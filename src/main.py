import numpy as np

from Camo_Worm import Camo_Worm, initialise_random_clew
from Drawing import Drawing
from image import prep_image


image = prep_image()

# Random Clew
clew = initialise_random_clew(100, image.shape, (40, 30, 1))
drawing = Drawing(image)
drawing.add_worms(clew)
drawing.show()

# Worm Example
worm1 = Camo_Worm(200, 100, 50, np.pi/6, 70, np.pi/3, 10, 0.8)
drawing1 = Drawing(image)
drawing1.add_worm_with_details(worm1)

# Another Worm Example
worm2 = Camo_Worm(350, 100, 300, 0, 70, -np.pi/2, 5, 0.8)
drawing2 = Drawing(image)
drawing2.add_worm_with_details(worm2)

