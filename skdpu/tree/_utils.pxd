# Authors: Sylvan Brocard
#
# License: MIT

import numpy as np
cimport numpy as np

from sklearn.tree._utils cimport safe_realloc
from ._splitter cimport RandomDpuSplitter
from sklearn.tree._splitter cimport SplitRecord

ctypedef np.npy_intp SIZE_t  # Type for indices and counters

cdef extern from "src/trees_common.h":
    int MAX_CLASSES  # if there's a bug here try enum: MAX_CLASSES

# =============================================================================
# Table data structure
# =============================================================================

# A record on the table for breadth-first tree growing
cdef struct SetRecord:
    SIZE_t leaf_index
    SIZE_t depth
    SIZE_t parent
    SIZE_t n_node_samples
    SIZE_t n_left
    SIZE_t n_right
    SIZE_t n_constant_features
    bint has_minmax
    bint has_evaluated
    bint is_leaf
    bint is_left
    bint first_seen
    double impurity
    double current_proxy_improvement
    SplitRecord current
    SplitRecord best
    SIZE_t features[MAX_CLASSES]
    SIZE_t constant_features[MAX_CLASSES]

    # splitter loop variables that we keep track of
    SIZE_t f_i
    SIZE_t f_j
    SIZE_t n_total_constants
    SIZE_t n_found_constants
    SIZE_t n_drawn_constants
    SIZE_t n_known_constants
    SIZE_t n_visited_features

    # criterion class attributes that we keep track of
    double weighted_n_node_samples
    double weighted_n_left
    double weighted_n_right

cdef class Set:
    cdef SIZE_t capacity
    cdef SIZE_t top
    cdef SetRecord * set_

    cdef bint is_empty(self) nogil
    cdef int push(self, SIZE_t n_node_samples, SIZE_t depth, SIZE_t parent, bint is_left,
                  double impurity, SIZE_t n_constant_features, SIZE_t leaf_index,
                  SetRecord * parent_record, SIZE_t n_features) nogil except -1
    cdef int remove(self, SIZE_t index) nogil
    cdef int prune_leaves(self) nogil
