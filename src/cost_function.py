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

    for i, worm in enumerate(clew):
        if i < 5: 
            worm.print_deets()
            print("Clew Size: ", len(clew))




        # Internal Knowledge
        # Add cost function here...


        # Group Knowledge (i.e. Distance)
        # Add cost function here...
        # for ii in range(i, len(clew)):


        # Environment Knowledge
        # Add cost function here...



        


