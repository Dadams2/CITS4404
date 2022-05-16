import numpy as np
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.bezier as mbezier
from sklearn.metrics.pairwise import euclidean_distances


Path = mpath.Path


class Camo_Worm:
    def __init__(self, x, y, r, theta, deviation_r, deviation_gamma, width, colour):
        self.x = x
        self.y = y
        self.r = r
        self.theta = theta
        self.dr = deviation_r
        self.dgamma = deviation_gamma
        self.width = width
        self.colour = colour
        p0 = [self.x - self.r * np.cos(self.theta), self.y - self.r * np.sin(self.theta)]
        p2 = [self.x + self.r * np.cos(self.theta), self.y + self.r * np.sin(self.theta)]
        p1 = [self.x + self.dr * np.cos(self.theta+self.dgamma), self.y + self.dr * np.sin(self.theta+self.dgamma)]
        self.bezier = mbezier.BezierSegment(np.array([p0, p1,p2]))

    def control_points (self):
        return self.bezier.control_points

    def path (self):
        return mpath.Path(self.control_points(), [Path.MOVETO, Path.CURVE3, Path.CURVE3])

    def patch (self):
        return mpatches.PathPatch(self.path(), fc='None', ec=str(self.colour), lw=self.width, capstyle='round')

    def intermediate_points(self, intervals=None):
        if intervals is None:
            intervals = max(3, int(np.ceil(self.r/8)))
        return self.bezier.point_at_t(np.linspace(0,1,intervals))

    def approx_length (self):
        intermediates = self.intermediate_points()
        eds = euclidean_distances(intermediates,intermediates)
        return np.sum(np.diag(eds,1))

    def colour_at_t(self, t, image):
        intermediates = np.int64(np.round(np.array(self.bezier.point_at_t(t)).reshape(-1,2)))
        colours = [image[point[0],point[1]] for point in intermediates]
        return(np.array(colours)/255)

    def centre_point(self):
        return (self.x, self.y)
        
    def print_deets(self):
        print("Worm's 8 Parameters:")
        print(f"Centre Point (x, y): ({self.x}, {self.y})")
        print("Radius (r)          : ", self.r)
        print("Angle (theta)       : ", self.theta)
        print("Radius (d)          : ", self.dr)
        print("Angle (gamma)       : ", self.dgamma)
        print("Width               : ", self.width)
        print("Colour              : ", self.colour)


def random_worm (imshape, init_params):
    """
        Example of a random worm. You may do this differently.
        centre points, angles and colour chosen from uniform distributions
        lengths chosen from normal distributions with two std parameters passed
        width chosen from gamma distribution with shape parameter 3 and scale passed
    """

    # Just a Random number GENERATOR that will be used later for "generating random numbers"
    rng = np.random.default_rng() 

    (radius_std, deviation_std, width_theta) = init_params
    (ylim, xlim) = imshape
    midx = xlim * rng.random()
    midy = ylim * rng.random()
    r = radius_std * np.abs(rng.standard_normal())
    theta = rng.random() * np.pi
    dr = deviation_std * np.abs(rng.standard_normal())
    dgamma = rng.random() * np.pi
    colour = rng.random()
    width = width_theta * rng.standard_gamma(3)
    return Camo_Worm(midx, midy, r, theta, dr, dgamma, width, colour)


def initialise_random_clew (size, imshape, init_params):
    """ Initialise a random clew """
    
    clew: list[Camo_Worm] = []

    for i in range(size):
        clew.append(random_worm(imshape, init_params))
    
    return clew
