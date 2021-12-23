# Authors: Sylvan Brocard
#
# License: MIT

import numpy as np
cimport numpy as np

from ._utils cimport Set
from ._utils cimport SetRecord

# =============================================================================
# Types and constants
# =============================================================================

cdef double INFINITY = np.inf

# Some handy constants (DpuTreeBuilder)
cdef int IS_FIRST = 1
cdef int IS_NOT_FIRST = 0
cdef int IS_LEFT = 1
cdef int IS_NOT_LEFT = 0

TREE_LEAF = -1
TREE_UNDEFINED = -2
cdef SIZE_t _TREE_LEAF = TREE_LEAF
cdef SIZE_t _TREE_UNDEFINED = TREE_UNDEFINED
cdef SIZE_t INITIAL_STACK_SIZE = 10

# DPU Breadth first builder ----------------------------------------------------------

cdef inline int _add_to_frontier(SetRecord * rec,
                                 Set frontier) nogil except -1:
    """Adds record ``rec`` to the active leaf set ``frontier``

    Returns -1 in case of failure to allocate memory (and raise MemoryError)
    or 0 otherwise.
    """
    return frontier.push(rec.node_id, rec.depth, rec.parent, rec.is_leaf, rec.impurity, rec.n_constant_features)

cdef class DpuTreeBuilder(TreeBuilder):
    """Build a decision tree in a breadth-first fashion in parallel"""

    def __cinit__(self, Splitter splitter, SIZE_t min_samples_split, SIZE_t min_samples_leaf, SIZE_t max_depth,
                  double min_impurity_decrease):
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease

    cpdef build(self, Tree tree, object X, np.ndarray y, np.ndarray sample_weight=None) nogil except -1:
        """Build a decision tree from the training set (X,y)"""

        # check input
        X, y, sample_weight = self._check_input(X, y, sample_weight)

        cdef DOUBLE_t * sample_weight_ptr = NULL

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

        cdef Set frontier = Set(INITIAL_STACK_SIZE)
        cdef SetRecord* record
        cdef SetRecord* split_node_left
        cdef SetRecord* split_node_right

        cdef SIZE_t n_node_samples = splitter.n_samples
        cdef SIZE_t frontier_length
        cdef bint has_minmax
        cdef bint has_evaluated
        cdef int rc = 0
        cdef Node* node

        with nogil:
            # add root to frontier
            rc = self._add_split_node(splitter, tree, 0, n_node_samples,
                                      INFINITY, IS_FIRST, IS_LEFT, NULL, 0,
                                      &split_node_left)
            if rc >= 0:
                rc = _add_to_frontier(&split_node_left, frontier)

            if rc == -1:
                with gil:
                    raise MemoryError()

            while not frontier.is_empty():
                # fill instruction list
                # TODO: add queue clear
                frontier_length = frontier.top
                for i_record in range(frontier_length):
                    record = frontier.set_[i_record]

                    node = &tree.nodes[record.node_id]
                    has_minmax = record.has_minmax
                    has_evaluated = record.has_evaluated

                    if not has_minmax:
                        # TODO: add minmax instruction to queue
                        rc = add_minmax_instruction()
                        if rc == -1:
                            break

                    elif not has_evaluated:
                        # TODO: add split evaluate instruction
                        rc = add_evaluate_instruction()
                        if rc == -1:
                            break

                    else:
                        # TODO: add split commit instruction
                        rc = add_commit_instruction()
                        if rc == -1:
                            break

                # execute instruction list on DPUs
                # TODO; add call to instruction set execution
                rc = execute_instructions()

                #parse and process DPU output
                for i_record in range(frontier_length):
                    record = &frontier.set_[i_record]

                    node = &tree.nodes[record.node_id]
                    has_minmax = record.has_minmax
                    has_evaluated = record.has_evaluated

                    if not has_minmax:
                        # TODO: parse minmax result
                        rc = update_minmax()
                        if rc == -1:
                            break
                        record.has_minmax = True

                    elif not has_evaluated:
                        # TODO: parse evaluate result
                        rc = update_evaluation()
                        if rc == -1:
                            break
                        if record.splitter.finished_evaluation():
                            record.has_evaluated = True

                    else:
                        # TODO: deal with commit result
                        # Compute left split node
                        rc = self._add_split_node(&split_node_left)
                        if rc == -1:
                            break

                        # tree.nodes may have changed
                        node = &tree.nodes[record.node_id]

                        # Compute right split node
                        rc = self._add_split_node(&split_node_right)
                        if rc == -1:
                            break

                        # Add nodes to queue
                        rc = _add_to_frontier(&split_node_left)
                        if rc == -1:
                            break

                        rc = _add_to_frontier(&split_node_right)
                        if rc == -1:
                            break

                frontier.prune_leaves()

            if rc >= 0:
                rc = tree._resize_c(tree.node_count)

        if rc == -1:
            raise MemoryError()

    cdef inline int _add_split_node(self) nogil except -1:
        pass