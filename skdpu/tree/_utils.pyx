# Authors: Sylvan Brocard
#
# License: MIT

from libc.stdlib cimport free
from libc.stdlib cimport malloc

# =============================================================================
# Table data structure
# =============================================================================

cdef class Set:
    """A data structure for traversing.

    Attributes
    ----------
    capacity : SIZE_t
        The elements the table can hold; if more added then ``self.set_``
        needs to be resized.

    top : SIZE_t
        The number of elements currently in the set.

    set_ : StackRecord pointer
        The set of records.
    """

    def __cinit__(self, SIZE_t capacity):
        self.capacity = capacity
        self.top = 0
        self.stack_ = <SetRecord*> malloc(capacity * sizeof(SetRecord))

    def __dealloc__(self):
        free(self.stack_)

    cdef bint is_empty(self) nogil:
        return self.top <= 0

    cdef int push(self, SIZE_t node_id, SIZE_t start, SIZE_t end, SIZE_t depth, SIZE_t parent, bint is_leaf,
                  double impurity, SIZE_t n_constant_features) nogil except -1:
        """Add an element to the table.
        
        Return -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        cdef SIZE_t top = self.top
        cdef SetRecord* set_ = NULL

        # Resize if capacity not sufficient
        if top >= self.capacity:
            self.capacity *= 2
            # Since safe_realloc can raise MemoryError, use `except -1`
            safe_realloc(&self.stack_, self.capacity)

        set_ = self.set_
        set_[top].node_id = node_id
        set_[top].start = start
        set_[top].end = end
        set_[top].depth = depth
        set_[top].parent = parent
        set_[top].is_leaf = is_leaf
        set_[top].impurity = impurity
        set_[top].n_constant_features = n_constant_features

        # Increment stack pointer
        self.top = top + 1
        return 0

    cdef int remove(self, SIZE_t index) nogil:
        """Removes an element from the set"""
        cdef SIZE_t top = self.top
        if index < top:
            self.set_[index] = self.set_[top]
        elif index > top:
            with gil:
                raise ValueError("Trying to remove wrong element from the frontier set.")

        self.top = top - 1
        return 0