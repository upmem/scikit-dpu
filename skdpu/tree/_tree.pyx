# Authors: Sylvan Brocard
#
# License: MIT

import numpy as np
cimport numpy as np

from ._utils cimport Set
from ._utils cimport SetRecord

from ._splitter cimport RandomDpuSplitter

cdef extern from "src/trees_common.h":
    int SPLIT_EVALUATE
    int SPLIT_COMMIT
    int SPLIT_MINMAX
    int MAX_NB_LEAF
    struct Command:
        UINT8_t type
        UINT8_t feature_index
        UINT16_t leaf_index
        DTYPE_t feature_threshold

cdef extern from "src/trees.h":
    ctypedef struct Params:
        pass
    struct CommandArray:
        UINT32_t nb_cmds
        Command cmds[MAX_NB_LEAF]
    struct CommandResults:
        pass
    void addCommmand(CommandArray * arr, Command cmd)
    void pushCommandArray(Params * p, CommandArray * arr)
    void syncCommandArrayResults(Params * p, CommandArray * cmd_arr, CommandResults * res);

# =============================================================================
# Types and constants
# =============================================================================

cdef double INFINITY = np.inf
cdef double EPSILON = np.finfo('double').eps

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
    return frontier.push(rec.n_node_samples, rec.depth, rec.parent, rec.is_left, rec.impurity, rec.n_constant_features,
                         rec.weighted_n_node_samples)

cdef class DpuTreeBuilder(TreeBuilder):
    """Build a decision tree in a breadth-first fashion in parallel"""

    def __cinit__(self, RandomDpuSplitter splitter, SIZE_t min_samples_split, SIZE_t min_samples_leaf, SIZE_t max_depth,
                  double min_impurity_decrease, Params * p):
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.p = p

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
        cdef RandomDpuSplitter splitter = self.splitter
        cdef SIZE_t max_depth = self.max_depth
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef SIZE_t min_samples_split = self.min_samples_split
        cdef double min_impurity_decrease = self.min_impurity_decrease

        cdef Params * p = self.p

        # Recursive partition (without actual recursion)
        splitter.init(X, y, sample_weight_ptr)

        cdef Set frontier = Set(INITIAL_STACK_SIZE)
        cdef SetRecord * record
        cdef SetRecord * split_node_left
        cdef SetRecord * split_node_right
        cdef SplitRecord split
        cdef SplitRecord * best

        cdef SIZE_t node_id
        cdef SIZE_t parent
        cdef SIZE_t depth
        cdef SIZE_t leaf_index
        cdef SIZE_t n_leaves = 0
        cdef SIZE_t n_node_samples = splitter.n_samples
        cdef double weighted_n_samples = n_node_samples * 1.0  # not supporting sample weights
        cdef double weighted_n_node_samples
        cdef SIZE_t frontier_length
        cdef SIZE_t n_constant_features
        cdef SIZE_t depth
        cdef double impurity = INFINITY
        cdef bint has_minmax
        cdef bint has_evaluated
        cdef bint is_leaf
        cdef bint is_left
        cdef bint first_seen
        cdef int rc = 0
        cdef Node * node
        cdef Command * command
        cdef CommandArray cmd_arr
        cdef CommandResults res

        cdef SIZE_t minmax_index
        cdef SIZE_t eval_index

        cdef SIZE_t * features
        cdef SIZE_t * constant_features

        with nogil:
            # add root to frontier
            rc = frontier.push(n_node_samples, 0, _TREE_UNDEFINED, 0, INFINITY, 0, weighted_n_samples)
            if rc == -1:
                with gil:
                    raise MemoryError()

            n_leaves += 1

            while not frontier.is_empty():
                # initializing the command array
                cmd_arr.nb_cmds = 0

                # fill instruction list
                frontier_length = frontier.top
                for i_record in range(frontier_length):
                    record = frontier.set_[i_record]

                    has_minmax = record.has_minmax
                    has_evaluated = record.has_evaluated
                    depth = record.depth
                    first_seen = record.first_seen
                    n_node_samples = record.n_node_samples
                    weighted_n_node_samples = record.weighted_n_node_samples
                    impurity = record.impurity
                    best = &record.best

                    if first_seen:
                        # check if the node is eligible for splitting
                        is_leaf = (depth >= max_depth or
                                   n_node_samples < min_samples_split or
                                   n_node_samples < 2 * min_samples_leaf or
                                   weighted_n_node_samples < 2 * min_weight_leaf)
                        if is_leaf:
                            record.has_evaluated = True
                            record.is_leaf = True
                        record.first_seen = False

                    # evaluating node
                    if not has_evaluated:
                        if not has_minmax:
                            rc = splitter.draw_feature(&record)
                            if rc == -1:
                                break

                            # draw_feature might have declared that we finished evaluating the node
                            has_evaluated = record.has_evaluated
                            if not has_evaluated:
                                rc = self.add_minmax_instruction(command, record, &cmd_arr)
                                if rc == -1:
                                    break

                        else:
                            rc = self.add_evaluate_instruction(command, record, &cmd_arr)
                            if rc == -1:
                                break

                    # finalizing node
                    if has_evaluated:
                        # checking if node should be split after computing its improvement
                        rc = splitter.impurity_improvement(impurity, best, &record)
                        if rc == -1:
                            break
                        is_leaf = (is_leaf or
                                   best.improvement + EPSILON < min_impurity_decrease)
                        record.is_leaf = is_leaf

                        if not is_leaf:
                            rc = self.add_commit_instruction(command, record, &cmd_arr)
                            if rc == -1:
                                break

                # execute instruction list on DPUs
                pushCommandArray(self.p, &cmd_arr)
                syncCommandArrayResults(self.p, &cmd_arr, &res)

                minmax_index = 0
                eval_index = 0
                # parse and process DPU output
                for i_record in range(frontier_length):
                    record = &frontier.set_[i_record]

                    has_minmax = record.has_minmax
                    has_evaluated = record.has_evaluated
                    is_leaf = record.is_leaf
                    is_left = record.is_left
                    parent = record.parent
                    leaf_index = record.leaf_index
                    depth = record.depth
                    n_constant_features = record.n_constant_features
                    impurity = record.impurity
                    best = &record.best
                    n_node_samples = record.n_node_samples
                    weighted_n_node_samples = record.weighted_n_node_samples

                    if not has_evaluated:
                        if not has_minmax:
                            rc = splitter.draw_threshold(&record, &res, minmax_index)
                            if rc == -1:
                                break
                            minmax_index += 1

                        else:
                            rc = splitter.update_evaluation(&record, &res, eval_index)
                            if rc == -1:
                                break
                            eval_index += 1

                    elif not is_leaf:
                        # adding node to the tree
                        node_id = tree._add_node(parent, is_left, is_leaf, best.feature,
                                                 best.threshold, impurity, n_node_samples,
                                                 weighted_n_node_samples)

                        record.is_leaf = True  # not actually a leaf, just marking for deletion

                        # add both children to the frontier

                        # left child gets leaf index of its parent
                        n_node_samples = record.n_left
                        impurity = record.best.impurity_left
                        # TODO: add array of feature index to SetRecord structure
                        rc = frontier.push(n_node_samples, depth + 1, node_id, True, impurity, n_constant_features,
                                           leaf_index, record, splitter.n_features)
                        if rc == -1:
                            break

                        n_node_samples = record.n_right
                        impurity = record.best.impurity_right
                        # right child gets first available leaf index
                        rc = frontier.push(n_node_samples, depth + 1, node_id, False, impurity, n_constant_features,
                                           n_leaves)
                        if rc == -1:
                            break
                        n_leaves += 1

                    else:
                        # TODO: add record to tree as leaf
                        node_id = tree._add_node(parent, is_left, True, _TREE_UNDEFINED,
                                                 TREE_UNDEFINED, impurity, n_node_samples,
                                                 weighted_n_node_samples)

                        node = tree.nodes[node_id]
                        node.left_child = _TREE_LEAF
                        node.right_child = _TREE_LEAF

                frontier.prune_leaves()

            if rc >= 0:
                rc = tree._resize_c(tree.node_count)

        if rc == -1:
            raise MemoryError()

    cdef inline int add_minmax_instruction(self, Command * command, SetRecord * record,
                                           CommandArray * cmd_arr) nogil except -1:
        """Adds a minmax instruction to the list."""
        command.type = SPLIT_MINMAX
        command.feature_index = record.current.feature
        command.leaf_index = record.leaf_index

        addCommmand(cmd_arr, *command)

        return 0

    cdef inline int add_evaluate_instruction(self, Command * command, SetRecord * record,
                                             CommandArray * cmd_arr) nogil except -1:
        """Adds a split evaluate instruction to the list."""
        command.type = SPLIT_EVALUATE
        command.feature_index = record.current.feature
        command.leaf_index = record.leaf_index
        command.feature_threshold = record.current.threshold

        addCommmand(cmd_arr, *command)

    cdef inline int add_commit_instruction(self, Command * command, SetRecord * record,
                                           CommandArray * cmd_arr) nogil except -1:
        """Adds a split commit instruction to the list."""
        command.type = SPLIT_COMMIT
        command.feature_index = record.best.feature
        command.leaf_index = record.leaf_index
        command.feature_threshold = record.best.threshold

        addCommmand(cmd_arr, *command)
