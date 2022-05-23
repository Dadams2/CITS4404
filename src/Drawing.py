import numpy as np
import imageio

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from Camo_Worm import Camo_Worm


# Global variables

IMAGE_DIR = 'images'
IMAGE_NAME='original'
MASK = [320, 560, 160, 880] # ymin ymax xmin xmax

# Read, crop and display image and stats

def crop(image, mask):
    h, w = np.shape(image)
    return image[max(mask[0],0):min(mask[1],h), max(mask[2],0):min(mask[3],w)]

def prep_image():
    imdir = IMAGE_DIR
    imname = IMAGE_NAME
    mask = MASK

    print("Image name (shape) (intensity max, min, mean, std)\n")
    image = np.flipud(crop(imageio.imread(imdir+'/'+imname+".png"), mask))
    print("{} {} ({}, {}, {}, {})".format(imname, np.shape(image), np.max(image), np.min(image), round(np.mean(image),1), round(np.std(image),1)))
    plt.imshow(image, vmin=0, vmax=255, cmap='gray', origin='lower') # use vmin and vmax to stop imshow from scaling
    plt.show()
    return image



class Drawing:
    def __init__ (self, image):
        self.fig, self.ax = plt.subplots()
        self.image = image
        self.im = self.ax.imshow(self.image, cmap='gray', origin='lower')

    def add_patches(self, patches):
        try:
            for patch in patches:
                self.ax.add_patch(patch)
        except TypeError:
            self.ax.add_patch(patches)

    def add_dots(self, points, radius=4, **kwargs):
        try:
            for point in points:
                self.ax.add_patch(mpatches.Circle((point[0],point[1]), radius, **kwargs))
        except TypeError:
            self.ax.add_patch(mpatches.Circle((points[0],points[1]), radius, **kwargs))

    def add_worms(self, worms):
        try:
            self.add_patches([w.patch() for w in worms])
        except TypeError:
            self.add_patches([worms.patch()])
    
    def show(self):
        plt.show()

    def add_title(self, title):
        plt.title(title)
    
    def save(self, path_to_save):
        plt.savefig(path_to_save)


    def add_worm_with_details(self, worm: Camo_Worm, save_path: str = None):
        """
        Simply adds a single worm with Centre Point, End Points, Intermediate Points, etc...
        """

        self.add_worms(worm)
        self.add_dots(worm.intermediate_points(8), radius=2, color='green')
        self.add_dots(worm.control_points(),color='orange')
        self.add_dots((worm.x, worm.y), color='blue')

        if save_path is None:
            self.show()
        else:
            self.save(path_to_save=f"{save_path}/bezier.png")
            self.show()

