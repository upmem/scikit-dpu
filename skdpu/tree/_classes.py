"""
This module gathers tree-based methods, including decision, regression and
randomized trees. Single and multi-output problems are both handled.
"""

# Authors: Sylvan Brocard
#
# License: MIT

from . import _splitter_dpu
from . import _criterion_dpu

from sklearn.tree import _splitter
from sklearn.tree import _classes
from sklearn.tree import _criterion
from sklearn.tree._classes import DecisionTreeClassifier

__all__ = [
    "DecisionTreeClassifier",
]

_classes.DENSE_SPLITTERS = DENSE_SPLITTERS = {"best": _splitter.BestSplitter, "random": _splitter.RandomSplitter,
                                              "random_dpu": _splitter_dpu.RandomDpuSplitter}
_classes.CRITERIA_CLF = CRITERIA_CLF = {"gini": _criterion.Gini, "entropy": _criterion.Entropy,
                                        "gini_dpu": _criterion_dpu.GiniDpu}
