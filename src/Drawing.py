import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from Camo_Worm import Camo_Worm


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

