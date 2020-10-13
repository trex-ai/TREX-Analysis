# The goal of this is to calculate the emerging nash equilibrium from a given simulation state
# ----------------------------------------------------------------------------------------------------------------------

# Get the data
    # acess a simulation database
    # import the profiles of all participants

# Construct an MPD
    # construct an MDP for each of the participants
    # import an action space for those participants
        # that action space must be discrete for now!
    # load the participants action history into the MDP

# until convergence
    # assessed by:
        # changes in return of each participant
        # changes in the policy of each participant (Wasserstein) --> how do we deal with multiple pi_*
        # an iteration ceiling

    # for each participant
        # (partially?) determine best response for participant wrt other participants
            # for no-EES sim:
                # backward pass of Value iteration
                # immediate convergence
                # construct pi_*
                    # if pi_* is multiple, randomly choose from best

    # do some metrics and display (see assessment criteria)

# --> we should now have a system that is converged to a Nash Equilibrium

# Save this as a separate table into the sim database (we know which one it is anyways)


