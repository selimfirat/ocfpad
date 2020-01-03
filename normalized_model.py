from pyod.models.base import BaseDetector
from pyod.utils import invert_order
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler


# A normalization detector_ constructed due to the feedback of Prof. Arashloo during presentation.
# This class Takes detector_ in constructor and performs normalization during training and prediction.
class NormalizedModel(BaseDetector):

    def __init__(self, detector_):

        super().__init__()
        self.detector_ = detector_

        self.normalizer = StandardScaler()

    def fit(self, X, **kwargs):

        self._classes = 2

        X = self.normalizer.fit_transform(X)

        self.detector_.fit(X)

        self.decision_scores_ = invert_order(
            self.detector_.decision_function(X))

        self._process_decision_scores()


    def decision_function(self, X, **kwargs):

        X = self.normalizer.transform(X)

        return self.detector_.decision_function(X)
