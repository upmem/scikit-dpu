# Authors: Sylvan Brocard
#
# License: MIT

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
    int MAX_CLASSES
    int MAX_NB_LEAF

cdef extern from "src/trees.h":
    ctypedef struct dpu_set:
        pass
    ctypedef struct Params:
        UINT64_t npoints
        UINT64_t npadded
        UINT64_t npointperdpu
        UINT32_t nfeatures
        UINT32_t ntargets
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
    struct CommandResults:
        INT32_t nb_gini
        INT32_t nb_minmax
        UINT32_t gini_cnt[MAX_NB_LEAF * 2 * MAX_CLASSES]
        DTYPE_t min_max[MAX_NB_LEAF * 2]

cdef class RandomDpuSplitter(Splitter):
    cdef SIZE_t * features  # Feature indices in X
    cdef SplitRecord best
    cdef SplitRecord current

    cdef int init_dpu(self,
        object X,
        const DOUBLE_t[:, ::1] y,
        DOUBLE_t* sample_weight,
        Params * p) except -1
    cdef int init_record(self, SetRecord * record) nogil
    cdef int draw_feature(self, SetRecord * record) nogil
    cdef int impurity_improvement(self, double impurity, SplitRecord * split, SetRecord * record) nogil
    cdef int draw_threshold(self, SetRecord * record, CommandResults * res, SIZE_t minmax_index) nogil
    cdef int update_evaluation(self, SetRecord * record, CommandResults * res, SIZE_t eval_index) nogil
