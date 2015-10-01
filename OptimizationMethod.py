class OptimizationMethod:
    """
    A general Optimization Method class. Other methods should
    inherit from this class and the common operations should be
    put here to avoid duplicated code.
    """

    def __init__(self,optimization_problem):
        self.problem = optimization_problem
