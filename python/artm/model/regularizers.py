class Regularizer(object):
    """
    Base class for regularizer.
    """
    pass


class PhiRegularizer(Regularizer):
    """
    Base class for regularizers of the Phi matrix.
    """
    pass


class SmoothSparsePhiRegularizer(PhiRegularizer):
    """
    Smoothing regularizer on Phi matrix. Useful for forcing topic to be more common (smooth) or
    more specific (sparse).

    R(Phi) = ...
    """
