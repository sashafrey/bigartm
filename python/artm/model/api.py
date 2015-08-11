from sklearn.base import BaseEstimator, TransformerMixin

from artm.wrapper import LibArtm

from .scorers import Scorer
from .regularizers import Regularizer


class ARTM(BaseEstimator, TransformerMixin):
    """
    ARTM learner.

    There are two ways to use ARTM: as a scikit-learn transformer or interactive by changing
    configuration of the model time to time.
    """


    def __init__(self, topics=10, namespaces=None, **master_component_config):
        """
        :param topics: number of topics or list of names or string with pattern
        :param namespaces: list of namespace names or string with pattern
        :return:
        """
        self.lib = LibArtm()

        self.regularizers = RegularizerSet()
        self.scorers = ScorerSet()

        raise NotImplementedError()

    def fit(self, X):
        raise NotImplementedError()

    def transform(self, X):
        raise NotImplementedError()



class RegularizerSet(object):
    pass


class ScorerSet(object):
    pass

