import numpy as np
from Camo_Worm import Camo_Worm


def costfn(clew: list[Camo_Worm]):
    """
    The overall cost function, which is split into Internal, Group and Environmental types of the information.

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

