import neural_ode

def test_solver():
    """
    make sure solver is solving correctly
    """

    assert neural_ode.solver.model([0,0],0,[[1,1],[1,1]]) == [[0.5, 0.5]], "Something is wrong with <solver.model>"