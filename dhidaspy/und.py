def undulator_k (bfield, period):
    """
    Get the K parameter for undulator

    bfield : float - field in (T)
    period : fload - period in (m)
    """

    return 93.36 * bfield * period
