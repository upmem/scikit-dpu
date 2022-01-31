# Authors: Sylvan Brocard
#
# License: MIT

import numpy as np
cimport numpy as np

from ._splitter cimport RandomDpuSplitter
from sklearn.tree._splitter cimport SplitRecord

cdef extern from "src/trees_common.h":
    enum: MAX_CLASSES

DEF MAX_FEATURES=68

ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

ctypedef fused realloc_ptr:
    # Add pointer types here as needed.
    (DTYPE_t*)
    (SIZE_t*)
    (unsigned char*)
    (DOUBLE_t*)
    (DOUBLE_t**)
    (SetRecord*)

cdef realloc_ptr safe_realloc(realloc_ptr* p, size_t nelems) nogil except *

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
    SIZE_t features[MAX_FEATURES]
    SIZE_t constant_features[MAX_FEATURES]
    double sum_total[MAX_CLASSES]
    double sum_left[MAX_CLASSES]
    double sum_right[MAX_CLASSES]

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
                  SetRecord * parent_record, SIZE_t n_features, SIZE_t n_classes) nogil except -1
    cdef int remove(self, SIZE_t index) nogil
    cdef int prune_leaves(self) nogil
