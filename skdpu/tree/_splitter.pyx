# Authors: Sylvan Brocard
#
# License: MIT

# Disclaimer: Part of this code is adapted from scikit-learn
# with the following license:
# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Noel Dawe <noel@dawe.me>
#          Satrajit Gosh <satrajit.ghosh@gmail.com>
#          Lars Buitinck
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#          Joel Nothman <joel.nothman@gmail.com>
#          Fares Hedayati <fares.hedayati@gmail.com>
#          Jacob Schreiber <jmschreiber91@gmail.com>
#
# License: BSD 3 clause

from time import perf_counter

import xxhash
from sklearn.tree._splitter cimport Splitter

from libc.string cimport memcpy
from libc.stdio cimport printf

import numpy as np
cimport numpy as np
np.import_array()

from . cimport _dimm
from . import _dimm
from . import _perfcounter

try:
    from importlib.resources import files, as_file
except ImportError:
    # Try backported to PY<39 `importlib_resources`.
    from importlib_resources import files, as_file  # noqa

from sklearn.tree._utils cimport rand_int
from sklearn.tree._utils cimport rand_uniform

cdef extern from "src/trees.h" nogil:
    void allocate(Params *p)
    void free_dpus(dpu_set allset)
    void reset_kernel(Params *p)
    void load_kernel(Params *p, const char *DPU_BINARY)
    DTYPE_t ** build_jagged_array(Params *p, DTYPE_t * features_flat)
    void populateDpu(Params *p, DTYPE_t **features, DTYPE_t *targets)

DEF CYTHON_DEBUG = 0

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

    cdef int init_dpu(self,
                  object X,
                  const DOUBLE_t[:, ::1] y,
                  DTYPE_t[:, ::1] y_float,
                  DOUBLE_t* sample_weight,
                  Params* p) except -1:
        """Initialize the splitter

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        cdef DTYPE_t ** features = NULL

        # Call parent init
        IF CYTHON_DEBUG >= 1:
            print("initializing base splitter")
        Splitter.init(self, X, y, sample_weight)
        IF CYTHON_DEBUG >= 1:
            print("updating parameters")
        p.npoints = self.n_samples
        p.nfeatures = self.n_features
        p.nclasses = (<GiniDpu>self.criterion).n_classes[0]

        tic = perf_counter()
        if not _dimm._allocated:
            IF CYTHON_DEBUG >= 1:
                print("allocating dpus")
            _dimm._requested_dpus = p.ndpu
            allocate(p)
            _dimm.allset = p.allset
            _dimm._allocated = True
            _dimm._nr_dpus = p.ndpu
        elif _dimm._requested_dpus != p.ndpu:
            # TODO: keep the dpu_set struct alive somewhere between runs
            # TODO: sort what should be kept in p and what should be kept in _dimm
            IF CYTHON_DEBUG >= 1:
                print("reallocating dpus")
            _dimm._requested_dpus = p.ndpu
            free_dpus(_dimm.allset)
            _dimm._data_id=None
            _dimm._kernel= ""
            allocate(p)
            _dimm.allset = p.allset
            _dimm._nr_dpus = p.ndpu
        else:
            IF CYTHON_DEBUG >= 1:
                print("dpus are already allocated")
            p.allset = _dimm.allset
            p.ndpu = _dimm._nr_dpus
        p.npointperdpu = p.npoints // p.ndpu

        if _dimm._kernel != "tree":
            IF CYTHON_DEBUG >= 1:
                print("loading kernel")
            kernel_bin = files("skdpu").joinpath("tree/src/dpu_programs/trees_dpu_kernel_v2")
            with as_file(kernel_bin) as DPU_BINARY:
                load_kernel(p, bytes(DPU_BINARY))
            _dimm._kernel = "tree"
            _dimm._data_id = None

        toc = perf_counter()
        _perfcounter.dpu_init_time = toc - tic

        h = xxhash.xxh3_64()  # data_id = id(X)
        h.update(X)
        data_id = h.digest()
        if _dimm._data_id != data_id:
            IF CYTHON_DEBUG >= 1:
                print("allocating X")
            self.X = X

            IF CYTHON_DEBUG >= 1:
                print("building jagged array")
            features = build_jagged_array(p, &self.X[0,0])
            # TODO: free the pointer array at some point

            IF CYTHON_DEBUG >= 1:
                print("populating dpu")
            tic = perf_counter()
            with nogil:
                populateDpu(p, features, &y_float[0,0])
            toc = perf_counter()
            # _perfcounter.cpu_pim_time = toc - tic
            _perfcounter.cpu_pim_time = p.cpu_pim_timer

            _dimm._data_id = data_id

        IF CYTHON_DEBUG >= 1:
            print("resetting kernel")
        reset_kernel(p)

        IF CYTHON_DEBUG >= 1:
            print("finished init_dpu")

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

        cdef SIZE_t f_i = record.f_i
        cdef SIZE_t f_j = -1

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

        IF CYTHON_DEBUG >= 2:
            printf("    drawing a feature :\n")

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

        IF CYTHON_DEBUG >= 2:
            printf("    weighted_n_node_samples: %f, weighted_n_samples: %f\n",
                   self.criterion.weighted_n_node_samples,
                   self.criterion.weighted_n_samples)
        if split.feature != -1:
            split.improvement = self.criterion.impurity_improvement(impurity, impurity_left, impurity_right)
        else:
            split.improvement = -INFINITY

        return 0

    cdef int draw_threshold(self, SetRecord * record, CommandResults * res, SIZE_t minmax_index, Params * p) nogil:
        """Draws a random threshold between recovered min and max"""
        cdef DTYPE_t min_feature_value
        cdef DTYPE_t max_feature_value
        cdef UINT32_t * random_state = &self.rand_r_state
        cdef SIZE_t * features = record.features
        cdef SIZE_t i

        IF CYTHON_DEBUG >= 2:
            printf("    Drawing threshold for leaf %ld and feature %ld\n", record.leaf_index, record.current.feature)

        min_feature_value = INFINITY
        max_feature_value = -INFINITY
        for i in range(p.ndpu):
            min_feature_value = min(min_feature_value, res[i].min_max[2 * minmax_index])
            max_feature_value = max(max_feature_value, res[i].min_max[2 * minmax_index + 1])

        IF CYTHON_DEBUG >= 2:
            printf("    min: %f, max: %f\n", min_feature_value, max_feature_value)

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
                record.current.threshold = min_feature_value

            IF CYTHON_DEBUG >= 2:
                printf("    drew %f\n", record.current.threshold)

        return 0

    cdef int update_evaluation(self, SetRecord * record, CommandResults * res, SIZE_t eval_index, Params * p) nogil:
        """Reads the split evaluation results sent by the DPU and updates if current is better than best"""
        cdef double current_proxy_improvement
        cdef SIZE_t n_left
        cdef SIZE_t n_right
        cdef int rc = 0
        cdef int i

        rc = (<GiniDpu>self.criterion).dpu_update(res, eval_index, p.ndpu, record.n_node_samples, &n_left, &n_right)
        if rc == -1:
            return -1

        current_proxy_improvement = self.criterion.proxy_impurity_improvement()

        IF CYTHON_DEBUG >= 2:
            printf("    leaf %ld: old improvement %f, new improvement %f\n", record.leaf_index, record.current_proxy_improvement, current_proxy_improvement)
        if current_proxy_improvement > record.current_proxy_improvement:
            record.current_proxy_improvement = current_proxy_improvement
            record.best = record.current
            record.weighted_n_left = self.criterion.weighted_n_left
            record.weighted_n_right = self.criterion.weighted_n_right
            record.n_left = n_left
            record.n_right = n_right
            IF CYTHON_DEBUG >= 2:
                printf("    left value: ")
                for i in range(p.nclasses):
                    printf("%f ", self.criterion.sum_left[i])
                printf("\n")
                printf("    right value: ")
                for i in range(p.nclasses):
                    printf("%f ", self.criterion.sum_right[i])
                printf("\n")
            memcpy(record.sum_left, self.criterion.sum_left, p.nclasses * sizeof(double))
            memcpy(record.sum_right, self.criterion.sum_right, p.nclasses * sizeof(double))
            self.criterion.children_impurity(&record.best.impurity_left, &record.best.impurity_right)

        record.has_minmax = False

        return 0

    cdef int node_split(self, double impurity, SplitRecord* split,
                        SIZE_t* n_constant_features) nogil except -1:
        """Finalizes the node split and computes the real impurity improvement"""
        pass
