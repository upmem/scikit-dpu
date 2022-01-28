# Authors: Sylvan Brocard
#
# License: MIT

import numpy as np
cimport numpy as np

from libc.stdio cimport printf
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

from ._utils cimport Set
from ._utils cimport SetRecord

from ._splitter cimport RandomDpuSplitter

cdef extern from "src/trees_common.h":
    int SPLIT_EVALUATE
    int SPLIT_COMMIT
    int SPLIT_MINMAX
    enum: MAX_NB_LEAF
    enum: MAX_CLASSES
    struct Command:
        UINT8_t type
        UINT8_t feature_index
        UINT16_t leaf_index
        DTYPE_t feature_threshold

cdef extern from "src/trees.h" nogil:
    struct CommandArray:
        UINT32_t nb_cmds
        Command cmds[MAX_NB_LEAF]
    struct CommandResults:
        UINT32_t nb_gini
        UINT32_t nb_minmax
        UINT32_t gini_cnt[MAX_NB_LEAF * 2 * MAX_CLASSES]
        DTYPE_t min_max[MAX_NB_LEAF * 2]
    void addCommand(CommandArray * arr, Command cmd)
    void pushCommandArray(Params * p, CommandArray * arr)
    void syncCommandArrayResults(Params * p, CommandArray * cmd_arr, CommandResults * res);

    void allocate(Params *p)
    void free_dpus(Params *p)
    void load_kernel(Params *p, const char *DPU_BINARY)
    void populateDpu(Params *p, float ** features)

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

# cdef inline int _add_to_frontier(SetRecord * rec,
#                                  Set frontier) nogil except -1:
#     """Adds record ``rec`` to the active leaf set ``frontier``
#
#     Returns -1 in case of failure to allocate memory (and raise MemoryError)
#     or 0 otherwise.
#     """
#     return frontier.push(rec.n_node_samples, rec.depth, rec.parent, rec.is_left, rec.impurity, rec.n_constant_features,
#                          rec.weighted_n_node_samples)

cdef class DpuTreeBuilder(TreeBuilder):
    """Build a decision tree in a breadth-first fashion in parallel"""

    def __cinit__(self, RandomDpuSplitter splitter, SIZE_t min_samples_split, SIZE_t min_samples_leaf,
                  double min_weight_leaf, SIZE_t max_depth,
                  double min_impurity_decrease, SIZE_t ndpu):
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.ndpu = ndpu
        print("initialized the builder")

    cpdef build(self, Tree tree, object X, np.ndarray y, np.ndarray sample_weight=None):
        """Build a decision tree from the training set (X,y)"""

        print("starting the build")

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

        cdef Params * p = &self.p

        print("initing dem DPU")

        # Recursive partition (without actual recursion)
        y_float = y.astype(np.float32)
        with nogil:
            p.ndpu = self.ndpu
        splitter.init_dpu(X, y, y_float, sample_weight_ptr, p)

        print("inited dat DPU")
        splitter.criterion.weighted_n_samples = splitter.weighted_n_samples
        splitter.criterion.n_samples = splitter.n_samples
        print(f"weighted_n_samples: {splitter.weighted_n_samples}")
        print(f"n_samples: {splitter.n_samples}")

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
        cdef double impurity = INFINITY
        cdef bint has_minmax
        cdef bint has_evaluated
        cdef bint is_leaf
        cdef bint is_left
        cdef bint first_seen
        cdef int rc = 0
        cdef Node * node = NULL
        cdef Command command
        cdef CommandArray cmd_arr
        cdef CommandResults * res

        cdef SIZE_t minmax_index
        cdef SIZE_t eval_index

        cdef SIZE_t * features
        cdef SIZE_t * constant_features

        print("\nentering nogil")
        with nogil:
            # initializing the results array
            res = <CommandResults *> malloc(p.ndpu * sizeof(CommandResults))

            # add root to frontier
            printf("adding root to frontier with %lu samples\n", n_node_samples)
            # TODO: compute root node impurity (idea: make an evaluate with a threshold at infinity)
            # TODO: compute root node value (use it to get impurity)
            rc = frontier.push(n_node_samples, 0, _TREE_UNDEFINED, False, 10.0, 0, 0, record, splitter.n_features,
                               p.nclasses)
            if rc == -1:
                with gil:
                    raise MemoryError()

            n_leaves += 1
            printf("added root to frontier\n")  # DEBUG

            while not frontier.is_empty():
                printf("\n\ndoing a new command layer\n")  #DEBUG
                # initializing the command array
                cmd_arr.nb_cmds = 0

                # fill instruction list
                frontier_length = frontier.top
                printf("frontier length : %ld\n", frontier_length)  #DEBUG
                for i_record in range(frontier_length):
                    record = &frontier.set_[i_record]

                    has_minmax = record.has_minmax
                    has_evaluated = record.has_evaluated
                    depth = record.depth
                    first_seen = record.first_seen
                    n_node_samples = record.n_node_samples
                    weighted_n_node_samples = record.weighted_n_node_samples
                    impurity = record.impurity
                    best = &record.best
                    is_leaf = record.is_leaf

                    printf("  record %ld, depth %ld, leaf index %ld\n", i_record, depth, record.leaf_index)

                    if first_seen:
                        # check if the node is eligible for splitting
                        is_leaf = (depth >= max_depth or
                                   n_node_samples < min_samples_split or
                                   n_node_samples < 2 * min_samples_leaf or
                                   weighted_n_node_samples < 2 * min_weight_leaf)
                        is_leaf = is_leaf or impurity <= EPSILON
                        printf("    leaf status: %d\n", is_leaf)  #DEBUG
                        if is_leaf:
                            record.has_evaluated = True
                            has_evaluated = True
                            record.is_leaf = True
                            is_leaf = True
                        record.first_seen = False

                    # evaluating node
                    if not has_evaluated:
                        if not has_minmax:
                            rc = splitter.draw_feature(record)
                            if rc == -1:
                                break

                            # draw_feature might have declared that we finished evaluating the node
                            has_evaluated = record.has_evaluated
                            if not has_evaluated:
                                rc = add_minmax_instruction(&command, record, &cmd_arr)
                                if rc == -1:
                                    break

                        else:
                            rc = add_evaluate_instruction(&command, record, &cmd_arr)
                            if rc == -1:
                                break

                    # finalizing node
                    if has_evaluated and not is_leaf:
                        printf("    we've evaluated\n")  #DEBUG
                        # checking if node should be split after computing its improvement
                        rc = splitter.impurity_improvement(impurity, best, record)
                        if rc == -1:
                            break
                        printf("    best improvement : %f\n", best.improvement)  # DEBUG
                        is_leaf = (is_leaf or
                                   best.improvement + EPSILON < min_impurity_decrease)
                        record.is_leaf = is_leaf

                        if not is_leaf:
                            rc = add_commit_instruction(&command, record, &cmd_arr)  # DEBUG
                            if rc == -1:
                                break

                # execute instruction list on DPUs
                if cmd_arr.nb_cmds:
                    printf("pushing command array\n")  # DEBUG
                    pushCommandArray(p, &cmd_arr)
                    printf("syncing results\n")  # DEBUG
                    syncCommandArrayResults(p, &cmd_arr, res)
                    printf("received results\n")  # DEBUG
                    printf("nb_gini = %i, nb_minmax = %i\n", res.nb_gini, res.nb_minmax)

                minmax_index = 0
                eval_index = 0
                printf("frontier length : %ld\n", frontier_length)  #DEBUG
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

                    printf("  record %ld, depth %ld, leaf_index %ld\n", i_record, depth, leaf_index)  # DEBUG

                    if not has_evaluated:
                        if not has_minmax:
                            rc = splitter.draw_threshold(record, res, minmax_index, p)
                            if rc == -1:
                                break
                            minmax_index += 1

                        else:
                            printf("    updating evaluation\n")  # DEBUG
                            rc = splitter.update_evaluation(record, res, eval_index, p)
                            if rc == -1:
                                break
                            eval_index += 1

                    elif not is_leaf:
                        # adding node to the tree
                        node_id = tree._add_node(parent, is_left, is_leaf, best.feature,
                                                 best.threshold, impurity, n_node_samples,
                                                 weighted_n_node_samples)
                        printf("    added node (split) %ld with parent %ld and %lu samples\n", node_id, parent,
                               n_node_samples)  # DEBUG

                        memcpy(splitter.criterion.sum_total, record.sum_total, p.nclasses * sizeof(double))
                        splitter.node_value(tree.value + node_id * tree.value_stride)

                        record.is_leaf = True  # not actually a leaf, just marking for deletion

                        # add both children to the frontier

                        # left child gets leaf index of its parent
                        n_node_samples = record.n_left
                        impurity = record.best.impurity_left
                        rc = frontier.push(n_node_samples, depth + 1, node_id, True, impurity, n_constant_features,
                                           leaf_index, record, splitter.n_features, p.nclasses)
                        if rc == -1:
                            break

                        n_node_samples = record.n_right
                        impurity = record.best.impurity_right
                        # right child gets first available leaf index
                        rc = frontier.push(n_node_samples, depth + 1, node_id, False, impurity, n_constant_features,
                                           n_leaves, record, splitter.n_features, p.nclasses)
                        if rc == -1:
                            break
                        n_leaves += 1

                    else:  # node is a leaf
                        node_id = tree._add_node(parent, is_left, True, _TREE_UNDEFINED,
                                                 _TREE_UNDEFINED, impurity, n_node_samples,
                                                 weighted_n_node_samples)
                        printf("    added node (leaf) %ld with parent %ld and %lu samples\n", node_id, parent,
                               n_node_samples)  # DEBUG

                        # TODO: refactor to make the copy once
                        memcpy(splitter.criterion.sum_total, record.sum_total, p.nclasses * sizeof(double))
                        printf("    sum_total : %f %f %f\n", splitter.criterion.sum_total[0],
                               splitter.criterion.sum_total[1], splitter.criterion.sum_total[2])  # DEBUG
                        splitter.node_value(tree.value + node_id * tree.value_stride)

                        node = &tree.nodes[node_id]
                        node.left_child = _TREE_LEAF
                        node.right_child = _TREE_LEAF

                frontier.prune_leaves()

            if rc >= 0:
                rc = tree._resize_c(tree.node_count)

            free(res)

        if rc == -1:
            raise MemoryError()

        free_dpus(p)

cdef inline int add_minmax_instruction(Command * command, SetRecord * record,
                                       CommandArray * cmd_arr) nogil except -1:
    """Adds a minmax instruction to the list."""
    command.type = SPLIT_MINMAX
    command.feature_index = record.current.feature
    command.leaf_index = record.leaf_index

    printf("    adding minmax on feature %d\n", command.feature_index)  #DEBUG

    addCommand(cmd_arr, command[0])

    return 0

cdef inline int add_evaluate_instruction(Command * command, SetRecord * record,
                                         CommandArray * cmd_arr) nogil except -1:
    """Adds a split evaluate instruction to the list."""
    command.type = SPLIT_EVALUATE
    command.feature_index = record.current.feature
    command.leaf_index = record.leaf_index
    command.feature_threshold = record.current.threshold

    printf("    adding evaluate on feature %d with threshold %f\n", command.feature_index,
           command.feature_threshold)  #DEBUG

    addCommand(cmd_arr, command[0])

cdef inline int add_commit_instruction(Command * command, SetRecord * record,
                                       CommandArray * cmd_arr) nogil except -1:
    """Adds a split commit instruction to the list."""
    command.type = SPLIT_COMMIT
    command.feature_index = record.best.feature
    command.leaf_index = record.leaf_index
    command.feature_threshold = record.best.threshold

    printf("    adding split on feature %d with threshold %f\n", command.feature_index,
           command.feature_threshold)  #DEBUG

    addCommand(cmd_arr, command[0])
