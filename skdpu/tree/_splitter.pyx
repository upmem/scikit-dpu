# Authors: Sylvan Brocard
#
# License: MIT

from sklearn.tree._criterion cimport Criterion
from sklearn.tree._splitter cimport Splitter

from libc.stdlib cimport free
from libc.stdlib cimport qsort
from libc.string cimport memcpy
from libc.string cimport memset

import numpy as np
cimport numpy as np
np.import_array()

from sklearn.tree._utils cimport log
from sklearn.tree._utils cimport rand_int
from sklearn.tree._utils cimport rand_uniform
from sklearn.tree._utils cimport RAND_R_MAX
from sklearn.tree._utils cimport safe_realloc

cdef extern from "src/trees.h":
    struct Params:
        pass
    void allocate(Params *p)
    void free_dpus(Params *p)
    void load_kernel(Params *p, const char *DPU_BINARY)
    void populateDpu(Params *p, float **features)
    void query_min_max(SIZE_t current_feature, DTYPE_t* min_feature_value, DTYPE_t* max_feature_value)
    void dpu_partition(DTYPE_t* current_threshold, SIZE_t* samples_left_leaf, SIZE_t* samples_right_leaf)

cdef double INFINITY = np.inf

# Mitigate precision differences between 32 bit and 64 bit
cdef DTYPE_t FEATURE_THRESHOLD = 1e-7

cdef inline void _init_split(SplitRecord* self, SIZE_t start_pos) nogil:
    self.impurity_left = INFINITY
    self.impurity_right = INFINITY
    self.pos = start_pos
    self.feature = 0
    self.threshold = 0.
    self.improvement = -INFINITY

cdef class RandomDpuSplitter(Splitter):
    """Splitter for finding the best random split on DPU."""

    cdef const DTYPE_t[:, :] X

    cdef SIZE_t n_total_samples

    cdef int init(self,
                  object X,
                  const DOUBLE_t[:, ::1] y,
                  DOUBLE_t* sample_weight) except -1:
        """Initialize the splitter

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """

        # Call parent init
        Splitter.init(self, X, y, sample_weight)

        self.X = X
        return 0

    # TODO: implement node_reset to put all the counters back to 0

    def __reduce__(self):
        return (RandomDpuSplitter, (self.criterion,
                                    self.max_features,
                                    self.min_samples_leaf,
                                    self.min_weight_leaf,
                                    self.random_state), self.__getstate__())

    cdef int draw_feature(self, SetRecord* record) nogil:
        """Draws a random feature to evaluate the next split on."""
        cdef SIZE_t * features = record.features
        cdef SIZE_t n_features = self.n_features

        cdef SIZE_t f_i = n_features
        cdef SIZE_t f_j

        cdef SIZE_t max_features = self.max_features
        cdef UINT32_t * random_state = &self.rand_r_state

        cdef SplitRecord * current = &record.current

        # Number of features discovered to be constant during the split search
        cdef SIZE_t n_found_constants = record.n_found_constants
        # Number of features known to be constant and drawn without replacement
        cdef SIZE_t n_drawn_constants = record.n_drawn_constants
        cdef SIZE_t n_known_constants = record.n_known_constants
        # n_total_constants = n_known_constants + n_found_constants
        cdef SIZE_t n_total_constants = record.n_total_constants
        cdef SIZE_t n_visited_features = record.n_visited_features

        cdef bint drew_nonconstant_feature = False

        # Sample up to max_features without replacement using a
        # Fisher-Yates-based algorithm (using the local variables `f_i` and
        # `f_j` to compute a permutation of the `features` array).
        #
        # Skip the CPU intensive evaluation of the impurity criterion for
        # features that were already detected as constant (hence not suitable
        # for good splitting) by ancestor nodes and save the information on
        # newly discovered constant features to spare computation on descendant
        # nodes.
        while (f_i > n_total_constants and  # Stop early if remaining features
                                            # are constant
                (n_visited_features < max_features or
                 # At least one drawn features must be non constant
                 n_visited_features <= n_found_constants + n_drawn_constants) and
                not drew_nonconstant_feature):
            n_visited_features += 1

            # Loop invariant: elements of features in
            # - [:n_drawn_constant[ holds drawn and known constant features;
            # - [n_drawn_constant:n_known_constant[ holds known constant
            #   features that haven't been drawn yet;
            # - [n_known_constant:n_total_constant[ holds newly found constant
            #   features;
            # - [n_total_constant:f_i[ holds features that haven't been drawn
            #   yet and aren't constant apriori.
            # - [f_i:n_features[ holds features that have been drawn
            #   and aren't constant.

            # Draw a feature at random
            f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
                           random_state)

            if f_j < n_known_constants:
                # f_j in the interval [n_drawn_constants, n_known_constants[
                features[n_drawn_constants], features[f_j] = features[f_j], features[n_drawn_constants]
                record.n_drawn_constants += 1

            else:
                # f_j in the interval [n_known_constants, f_i - n_found_constants[
                f_j += n_found_constants
                # f_j in the interval [n_total_constants, f_i[

                current.feature = features[f_j]
                drew_nonconstant_feature = True

        if not drew_nonconstant_feature:
            record.has_evaluated = True
            record.n_constant_features = record.n_total_constants

        record.f_j = f_j
        record.n_visited_features = n_visited_features

        return 0

    cdef int impurity_improvement(self, double impurity, SplitRecord * split, SetRecord * record) nogil:
        """Computes the impurity improvement of a given split"""
        cdef double impurity_left = split.impurity_left
        cdef double impurity_right = split.impurity_right

        self.criterion.weighted_n_left = record.weighted_n_left
        self.criterion.weighted_n_right = record.weighted_n_right
        self.criterion.weighted_n_node_samples = record.weighted_n_node_samples

        split.improvement = self.criterion.impurity_improvement(impurity, impurity_left, impurity_right)

        return 0

    cdef int draw_threshold(self, SetRecord * record, CommandResults * res, SIZE_t minmax_index) nogil:
        """Draws a random threshold between recovered min and max"""
        cdef DTYPE_t min_feature_value
        cdef DTYPE_t max_feature_value
        cdef UINT32_t * random_state = &self.rand_r_state
        cdef SIZE_t * features = record.features

        min_feature_value = res.min_max[2 * minmax_index]
        max_feature_value = res.min_max[2 * minmax_index + 1]

        if max_feature_value <= min_feature_value + FEATURE_THRESHOLD:
            features[record.f_j], features[record.n_total_constants] = (
                features[record.n_total_constants], record.current.feature)

            record.n_total_constants += 1
            record.n_found_constants += 1

        else:
            record.f_i -= 1
            features[record.f_i], features[record.f_j] = features[record.f_j], features[record.f_i]

            # Draw a random threshold
            record.current.threshold = rand_uniform(min_feature_value, max_feature_value, random_state)
            record.has_minmax = True

            if record.current.threshold == max_feature_value:
                record.threshold = min_feature_value

        return 0

    cdef int update_evaluation(self, SetRecord * record, CommandResults * res, SIZE_t eval_index) nogil:
        """Reads the split evaluation results sent by the DPU and updates if current is better than best"""
        (<GiniDpu>self.criterion).dpu_update(res, eval_index)
        cdef double current_proxy_improvement

        current_proxy_improvement = self.criterion.proxy_impurity_improvement()

        if current_proxy_improvement > record.current_proxy_improvement:
            record.current_proxy_improvement = current_proxy_improvement
            record.best = record.current
            record.weighted_n_left = self.criterion.weighted_n_left
            record.weighted_n_right = self.criterion.weighted_n_right

        return 0

    cdef int node_split(self, double impurity, SplitRecord* split,
                        SIZE_t* n_constant_features) nogil except -1:
        """Find the best random split on node samples[start:end]

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        # Draw random splits and pick the best
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end

        cdef SIZE_t* features = self.features
        cdef SIZE_t* constant_features = self.constant_features
        cdef SIZE_t n_features = self.n_features

        cdef DTYPE_t* Xf = self.feature_values
        cdef SIZE_t max_features = self.max_features
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef UINT32_t* random_state = &self.rand_r_state

        cdef SplitRecord best, current
        cdef double current_proxy_improvement = - INFINITY
        cdef double best_proxy_improvement = - INFINITY

        cdef SIZE_t f_i = n_features
        cdef SIZE_t f_j
        cdef SIZE_t p
        cdef SIZE_t partition_end
        cdef SIZE_t feature_stride
        # Number of features discovered to be constant during the split search
        cdef SIZE_t n_found_constants = 0
        # Number of features known to be constant and drawn without replacement
        cdef SIZE_t n_drawn_constants = 0
        cdef SIZE_t n_known_constants = n_constant_features[0]
        # n_total_constants = n_known_constants + n_found_constants
        cdef SIZE_t n_total_constants = n_known_constants
        cdef SIZE_t n_visited_features = 0
        cdef DTYPE_t min_feature_value
        cdef DTYPE_t max_feature_value
        cdef DTYPE_t current_feature_value

        cdef SIZE_t samples_left_leaf
        cdef SIZE_t samples_right_leaf

        _init_split(&best, end)

        # Sample up to max_features without replacement using a
        # Fisher-Yates-based algorithm (using the local variables `f_i` and
        # `f_j` to compute a permutation of the `features` array).
        #
        # Skip the CPU intensive evaluation of the impurity criterion for
        # features that were already detected as constant (hence not suitable
        # for good splitting) by ancestor nodes and save the information on
        # newly discovered constant features to spare computation on descendant
        # nodes.
        while (f_i > n_total_constants and  # Stop early if remaining features
                                            # are constant
                (n_visited_features < max_features or
                 # At least one drawn features must be non constant
                 n_visited_features <= n_found_constants + n_drawn_constants)):
            n_visited_features += 1

            # Loop invariant: elements of features in
            # - [:n_drawn_constant[ holds drawn and known constant features;
            # - [n_drawn_constant:n_known_constant[ holds known constant
            #   features that haven't been drawn yet;
            # - [n_known_constant:n_total_constant[ holds newly found constant
            #   features;
            # - [n_total_constant:f_i[ holds features that haven't been drawn
            #   yet and aren't constant apriori.
            # - [f_i:n_features[ holds features that have been drawn
            #   and aren't constant.

            # Draw a feature at random
            f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
                           random_state)

            if f_j < n_known_constants:
                # f_j in the interval [n_drawn_constants, n_known_constants[
                features[n_drawn_constants], features[f_j] = features[f_j], features[n_drawn_constants]
                n_drawn_constants += 1

            else:
                # f_j in the interval [n_known_constants, f_i - n_found_constants[
                f_j += n_found_constants
                # f_j in the interval [n_total_constants, f_i[

                current.feature = features[f_j]

                # Find min, max
                # min_feature_value = self.X[samples[start], current.feature]
                # max_feature_value = min_feature_value
                # Xf[start] = min_feature_value

                # for p in range(start + 1, end):
                #     current_feature_value = self.X[samples[p], current.feature]
                #     Xf[p] = current_feature_value

                #     if current_feature_value < min_feature_value:
                #         min_feature_value = current_feature_value
                #     elif current_feature_value > max_feature_value:
                #         max_feature_value = current_feature_value

                query_min_max(current.feature, &min_feature_value, &max_feature_value)

                if max_feature_value <= min_feature_value + FEATURE_THRESHOLD:
                    features[f_j], features[n_total_constants] = features[n_total_constants], current.feature

                    n_found_constants += 1
                    n_total_constants += 1

                else:
                    f_i -= 1
                    features[f_i], features[f_j] = features[f_j], features[f_i]

                    # Draw a random threshold
                    current.threshold = rand_uniform(min_feature_value,
                                                     max_feature_value,
                                                     random_state)

                    if current.threshold == max_feature_value:
                        current.threshold = min_feature_value

                    # Partition
                    # p, partition_end = start, end
                    # while p < partition_end:
                    #     if Xf[p] <= current.threshold:
                    #         p += 1
                    #     else:
                    #         partition_end -= 1

                    #         Xf[p], Xf[partition_end] = Xf[partition_end], Xf[p]
                    #         samples[p], samples[partition_end] = samples[partition_end], samples[p]

                    # current.pos = partition_end

                    dpu_partition(current.threshold, &samples_left_leaf, &samples_right_leaf)

                    # Reject if min_samples_leaf is not guaranteed
                    # if (((current.pos - start) < min_samples_leaf) or
                    #         ((end - current.pos) < min_samples_leaf)):
                    #     continue
                    if ((samples_left_leaf < min_samples_leaf) or
                            (samples_right_leaf < min_samples_leaf)):
                        continue

                    # Evaluate split
                    self.criterion.reset()
                    self.criterion.update(current.pos)

                    # Reject if min_weight_leaf is not satisfied
                    # if ((self.criterion.weighted_n_left < min_weight_leaf) or
                    #         (self.criterion.weighted_n_right < min_weight_leaf)):
                    #     continue

                    current_proxy_improvement = self.criterion.proxy_impurity_improvement()

                    if current_proxy_improvement > best_proxy_improvement:
                        best_proxy_improvement = current_proxy_improvement
                        best = current  # copy

        # Reorganize into samples[start:best.pos] + samples[best.pos:end]
        if best.pos < end:
            if current.feature != best.feature:
                # p, partition_end = start, end

                # while p < partition_end:
                #     if self.X[samples[p], best.feature] <= best.threshold:
                #         p += 1
                #     else:
                #         partition_end -= 1

                #         samples[p], samples[partition_end] = samples[partition_end], samples[p]
                dpu_partition(best.threshold, &samples_left_leaf, &samples_right_leaf)

            self.criterion.reset()
            self.criterion.update(best.pos)
            self.criterion.children_impurity(&best.impurity_left,
                                             &best.impurity_right)
            best.improvement = self.criterion.impurity_improvement(
                impurity, best.impurity_left, best.impurity_right)

        # Respect invariant for constant features: the original order of
        # element in features[:n_known_constants] must be preserved for sibling
        # and child nodes
        memcpy(features, constant_features, sizeof(SIZE_t) * n_known_constants)

        # Copy newly found constant features
        memcpy(constant_features + n_known_constants,
               features + n_known_constants,
               sizeof(SIZE_t) * n_found_constants)

        # Return values
        split[0] = best
        n_constant_features[0] = n_total_constants
        return 0
