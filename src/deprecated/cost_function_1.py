import numpy as np
from Camo_Worm import Camo_Worm


def costfn(clew: list[Camo_Worm]):
    """
    The overall cost function, which is split into Internal, Group and Environmental types of the information.
    Parameters
    ---
        clew: list
            A clew of worms
        worm_idx: int
            The index of the worm within the clew
        w_internal: float
            Weight for internal
        w_dist: float
            Weight for distance
    
    Returns
    ---
    float
        The score of the worm
    Overall Formula:
        Cost Function = SUM(iW.iC + gW.gC + eW.eC)
            iW - Weight/Tuning parameter for Internal Knowledge
            gW - Weight/Tuning parameter for Group Knowledge
            eW - Weight/Tuning parameter for Environment Knowledge
            iC - Cost Function for Internal Knowledge
            gC - Cost Function for Group Knowledge
            eC - Cost Function for Environment Knowledge
    Performance:
        Cost Function could be in single function, to avoid repetition of loops.
    """

    # Internal Knowledge
    # Add cost function here...
    internal_score = costfn_internal(clew, 200, 1)

    # Group Knowledge (i.e. Distance)
    # Add cost function here...


    distance_score: list[float] = [0.0] * len(clew)

    # Calculate the Euclidean distance using NumPy for each point
    # Adding the inverse of the distance to the sum
    # Complexity: (n^2)/2
    for i in range(0, len(clew)):
        for ii in range(i+1, len(clew)):
            if i is not ii:

                # Centre Point of the two worms
                point1 = np.array(clew[i].centre_point())
                point2 = np.array(clew[ii].centre_point())

                # Euclidean distance using NumPy
                eu_distance = np.linalg.norm(point1 - point2)

                # Adding to the sum for the two worms
                distance_score[i] += 1 / eu_distance
                distance_score[ii] += 1 / eu_distance


        # Environment Knowledge
        # Add cost function here...


# Basic internal drivers for cost function
#   Worm prefers to grow larger
#   Worm prefers to be straighter
def costfn_internal(clew: list, w_size: float, w_curve: float):
    """
    Evaluates internal drivers aspect of the cost function.

    Parameters
    ---
    clew : list
        A list of `Camo_Worm` objects.
    
    w_size: float
        The weighting to apply to worm size
    
    w_curve: float
        The weighting to apply to worm curvature
    
    Returns
    ---
    float
        The evaluation of the `clew` of `Camo_Worms` for internal drivers.
    """
    total_score = 0

    
    # Since this is operating on internal features we will consider
    # each worm individually
    for worm in clew:
        # larger worm = higher score
        # using length of the centre line for this so that curvature
        # isn't considered in the size of the worm
        # Inverse so larger worm is smaller penalty
        # [TODO] Figure out some sort of way to weight this as equal to curvature
        size_score = (1 / (2 * worm.r * worm.width))
        # straighter worm = higher score
        c_penalty = worm.dr

        total_score += size_score * w_size + c_penalty * w_curve
    
    return total_score