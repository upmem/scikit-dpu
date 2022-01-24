# Authors: Sylvan Brocard
#
# License: MIT

import numpy as np

from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libc.string cimport memcpy
from libc.stdlib cimport realloc
from libc.stdio cimport printf

TREE_UNDEFINED = -2
cdef SIZE_t _TREE_UNDEFINED = TREE_UNDEFINED
cdef double INFINITY = np.inf

# =============================================================================
# Table data structure
# =============================================================================

cdef realloc_ptr safe_realloc(realloc_ptr* p, size_t nelems) nogil except *:
    # sizeof(realloc_ptr[0]) would be more like idiomatic C, but causes Cython
    # 0.20.1 to crash.
    cdef size_t nbytes = nelems * sizeof(p[0][0])
    if nbytes / sizeof(p[0][0]) != nelems:
        # Overflow in the multiplication
        with gil:
            raise MemoryError("could not allocate (%d * %d) bytes"
                              % (nelems, sizeof(p[0][0])))
    cdef realloc_ptr tmp = <realloc_ptr>realloc(p[0], nbytes)
    if tmp == NULL:
        with gil:
            raise MemoryError("could not allocate %d bytes" % nbytes)

    p[0] = tmp
    return tmp  # for convenience

cdef class Set:
    """A data structure for traversing.

    Attributes
    ----------
    capacity : SIZE_t
        The elements the table can hold; if more added then ``self.set_``
        needs to be resized.

    top : SIZE_t
        The number of elements currently in the set.

    set_ : SetRecord pointer
        The set of records.
    """

    def __cinit__(self, SIZE_t capacity):
        self.capacity = capacity
        self.top = 0
        self.set_ = <SetRecord*> malloc(capacity * sizeof(SetRecord))

    def __dealloc__(self):
        free(self.set_)

    cdef bint is_empty(self) nogil:
        return self.top <= 0

    cdef int push(self, SIZE_t n_node_samples, SIZE_t depth, SIZE_t parent, bint is_left,
                  double impurity, SIZE_t n_constant_features, SIZE_t leaf_index,
                  SetRecord * parent_record, SIZE_t n_features) nogil except -1:
        """Add an element to the table.
        
        Return -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        printf("pushing an element\n") # DEBUG
        cdef SIZE_t top = self.top
        cdef SetRecord * top_record = NULL

        cdef SIZE_t i

        # Resize if capacity not sufficient
        if top >= self.capacity:
            printf("resizing the frontier") # DEBUG
            self.capacity *= 2
            # Since safe_realloc can raise MemoryError, use `except -1`
            safe_realloc(&self.set_, self.capacity)

        top_record = &self.set_[top]
        printf("now looking at top record at index %d\n", top)
        top_record.leaf_index = leaf_index
        top_record.depth = depth
        top_record.parent = parent
        top_record.is_left = is_left
        top_record.is_leaf = False
        top_record.impurity = impurity
        printf("impurity = %f\n", impurity)
        top_record.n_constant_features = n_constant_features
        top_record.n_node_samples = n_node_samples
        top_record.weighted_n_node_samples = n_node_samples  # no support for non-unity weights for DPU trees
        top_record.first_seen = True
        top_record.has_evaluated = False
        top_record.current_proxy_improvement = -INFINITY

        # initializing loop variables
        top_record.n_found_constants = 0
        top_record.n_drawn_constants = 0
        top_record.n_known_constants = n_constant_features
        top_record.n_total_constants = n_constant_features
        top_record.n_visited_features = 0
        top_record.f_i = n_features

        if not parent == _TREE_UNDEFINED:
            # TODO: optimize this copy, or change the structure to not move records in it
            memcpy(top_record.features, parent_record.features, sizeof(SIZE_t) * n_features)
            memcpy(top_record.constant_features, parent_record.constant_features, sizeof(SIZE_t) * n_features)

            # Respect invariant for constant features: the original order of
            # element in features[:n_known_constants] must be preserved for sibling
            # and child nodes
            memcpy(top_record.features, parent_record.constant_features, sizeof(SIZE_t) * parent_record.n_known_constants)
            # Copy newly found constant features
            memcpy(top_record.constant_features + parent_record.n_known_constants,
                   parent_record.features + parent_record.n_known_constants,
                   sizeof(SIZE_t) * parent_record.n_found_constants)
        else: # current node is the root
            for i in range(n_features):
                top_record.features[i] = i

        # Increment set pointer
        self.top = top + 1
        printf("done pushing\n") # DEBUG
        return 0

    cdef int remove(self, SIZE_t index) nogil:
        """Removes an element from the set"""
        cdef SIZE_t top = self.top

        top -= 1
        self.top = top

        if index < top:
            printf("removing element %i with depth %i and is_left %i\n", index, self.set_[index].depth, self.set_[index].is_left)
            printf("replacing with record at %i with depth %i and is_left %i\n", top, self.set_[top].depth, self.set_[top].is_left)
            self.set_[index] = self.set_[top]
            printf("replaced at %i by record with depth %i and is_left %i\n", index, self.set_[index].depth, self.set_[index].is_left)
        elif index > top:
            with gil:
                raise ValueError("Trying to remove wrong element from the frontier set.")

        return 0

    cdef int prune_leaves(self) nogil:
        """Removes all leaves from the set"""
        cdef SIZE_t top = self.top
        cdef SIZE_t i_record = 0
        while i_record < top:
            if self.set_[i_record].is_leaf:
                self.remove(i_record)
                top -= 1
            else:
                i_record += 1
        return 0
