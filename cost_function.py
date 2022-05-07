from Camo_Worm import Camo_Worm


def print_worm_deets(worm: Camo_Worm):
    print("Worm's 8 Parameters:")
    print(f"Centre Point (x, y): ({worm.x}, {worm.y})")
    print("Radius (r)          : ", worm.r)
    print("Angle (theta)       : ", worm.theta)
    print("Radius (d)          : ", worm.dr)
    print("Angle (gamma)       : ", worm.dgamma)
    print("Width               : ", worm.width)
    print("Colour              : ", worm.colour)


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

    for worm in clew:
        print_worm_deets(worm)


        # Internal Knowledge
        # Add cost function here...


        # Group Knowledge
        # Add cost function here...


        # Environment Knowledge
        # Add cost function here...



        


