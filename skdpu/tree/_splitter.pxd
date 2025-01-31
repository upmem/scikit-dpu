# Authors: Sylvan Brocard
#
# License: MIT

# Disclaimer: Part of this code is adapted from scikit-learn
# with the following license:
# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Joel Nothman <joel.nothman@gmail.com>
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#          Jacob Schreiber <jmschreiber91@gmail.com>
#
# License: BSD 3 clause

from sklearn.tree._tree cimport DTYPE_t  # Type of X
from sklearn.tree._tree cimport DOUBLE_t  # Type of y, sample_weight
from sklearn.tree._tree cimport SIZE_t  # Type for indices and counters
from sklearn.tree._tree cimport INT32_t  # Signed 32 bit integer
from sklearn.tree._tree cimport UINT32_t  # Unsigned 32 bit integer

from sklearn.tree._splitter cimport SplitRecord
from sklearn.tree._splitter cimport Splitter

from ._utils cimport SetRecord
from ._criterion cimport GiniDpu

from ._tree cimport UINT64_t

cdef extern from "src/trees_common.h":
    enum: MAX_CLASSES
    enum: MAX_NB_LEAF

cdef extern from "src/trees.h":
    ctypedef struct dpu_set:
        pass
    ctypedef struct Params:
        UINT64_t npoints
        UINT64_t npadded
        UINT64_t npointperdpu
        UINT32_t nfeatures
        UINT32_t ntargets
        UINT32_t nclasses
        DTYPE_t scale_factor
        DTYPE_t threshold
        DTYPE_t * mean
        int isOutput
        int nloops
        int max_iter
        UINT32_t ndpu
        dpu_set allset
        int from_file
        int verbose
        double dpu_timer
        double cpu_pim_timer
    struct CommandResults:
        INT32_t nb_gini
        INT32_t nb_minmax
        UINT32_t gini_cnt[MAX_NB_LEAF * 2 * MAX_CLASSES]
        DTYPE_t min_max[MAX_NB_LEAF * 2]

cdef class RandomDpuSplitter(Splitter):
    cdef SplitRecord best
    cdef SplitRecord current
    cdef DTYPE_t[:, :] X
    cdef SIZE_t n_total_samples

    cdef int init_dpu(self,
        object X,
        const DOUBLE_t[:, ::1] y,
        DTYPE_t[:, ::1] y_float,
        DOUBLE_t* sample_weight,
        Params * p) except -1
    cdef int draw_feature(self, SetRecord * record) nogil
    cdef int impurity_improvement(self, double impurity, SplitRecord * split, SetRecord * record) nogil
    cdef int draw_threshold(self, SetRecord * record, CommandResults * res, SIZE_t minmax_index, Params * p) nogil
    cdef int update_evaluation(self, SetRecord * record, CommandResults * res, SIZE_t eval_index, Params * p) nogil
    cdef int node_split(self, double impurity, SplitRecord* split,
                        SIZE_t* n_constant_features) nogil except -1
