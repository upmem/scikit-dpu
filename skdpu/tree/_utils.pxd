# Authors: Sylvan Brocard
#
# License: MIT

import numpy as np
cimport numpy as np

from sklearn.tree._utils cimport safe_realloc

ctypedef np.npy_intp SIZE_t  # Type for indices and counters

# =============================================================================
# Table data structure
# =============================================================================

# A record on the table for breadth-first tree growing
cdef struct SetRecord:
    SIZE_t node_id
    SIZE_t start
    SIZE_t end
    SIZE_t depth
    SIZE_t parent
    bint is_leaf
    double impurity
    SIZE_t n_constant_features

cdef class Set:
    cdef SIZE_t capacity
    cdef SIZE_t top
    cdef SetRecord * set_

    cdef bint is_empty(self) nogil
    cdef int push(self, SIZE_t node_id, SIZE_t start, SIZE_t end, SIZE_t depth, SIZE_t parent, bint is_leaf,
                  double impurity, SIZE_t n_constant_features) nogil except -1
    cdef int remove(self, SIZE_t index) nogil
