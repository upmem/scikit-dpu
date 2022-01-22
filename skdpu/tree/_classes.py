"""
This module gathers tree-based methods, including decision, regression and
randomized trees. Single and multi-output problems are both handled.
"""

# Authors: Sylvan Brocard
#
# License: MIT



from sklearn.tree import _splitter
from sklearn.tree import _classes
from sklearn.tree import _criterion
from sklearn.tree._classes import DecisionTreeClassifier

from ._splitter import RandomDpuSplitter
from ._criterion import GiniDpu

__all__ = [
    "DecisionTreeClassifier",
]

_classes.DENSE_SPLITTERS = DENSE_SPLITTERS = {"best": _splitter.BestSplitter, "random": _splitter.RandomSplitter,
                                              "random_dpu": RandomDpuSplitter}
_classes.CRITERIA_CLF = CRITERIA_CLF = {"gini": _criterion.Gini, "entropy": _criterion.Entropy,
                                        "gini_dpu": GiniDpu}
