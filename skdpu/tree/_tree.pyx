# Authors: Sylvan Brocard
#
# License: MIT

import numpy as np
cimport numpy as np

cdef class DpuTreeBuilder(TreeBuilder):
    """Build a decision tree in a breadth-first fashion in parallel"""

    def __cinit__(self, Splitter splitter, SIZE_t min_samples_split, SIZE_t min_samples_leaf, SIZE_t max_depth,
                  double min_impurity_decrease):
        self.splitter = splitter
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease

    cpdef build(self, Tree tree, object X, np.ndarray y, np.ndarray sample_weight=None):
        """Build a decision tree from the training set (X,y)"""

        # check input
        X, y, sample_weight = self._check_input(X, y, sample_weight)

        cdef DOUBLE_t* sample_weight_ptr = NULL

        # Initial capacity
        cdef int init_capacity

        if tree.max_depth <= 10:
            init_capacity = (2 ** (tree.max_depth + 1)) - 1
        else:
            init_capacity = 2047

        tree._resize(init_capacity)

        # Parameters
        cdef Splitter splitter = self.splitter
        cdef SIZE_t max_depth = self.max_depth
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef SIZE_t min_samples_split = self.min_samples_split
        cdef double min_impurity_decrease = self.min_impurity_decrease

        # Recursive partition (without actual recursion)
        splitter.init(X, y, sample_weight_ptr)

        with nogil:
            # build the tree
            pass
