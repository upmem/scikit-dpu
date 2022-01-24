# Authors: Sylvan Brocard
#
# License: MIT

from sklearn.tree._tree cimport DTYPE_t  # Type of X
from sklearn.tree._tree cimport DOUBLE_t  # Type of y, sample_weight
from sklearn.tree._tree cimport SIZE_t  # Type for indices and counters
from sklearn.tree._tree cimport INT32_t  # Signed 32 bit integer
from sklearn.tree._tree cimport UINT32_t  # Unsigned 32 bit integer

from sklearn.tree._criterion cimport Criterion

from ._utils cimport SetRecord

cdef extern from "src/trees_common.h":
    enum: MAX_CLASSES
    enum: MAX_NB_LEAF

cdef extern from "src/trees.h":
    struct CommandResults:
        INT32_t nb_gini
        INT32_t nb_minmax
        UINT32_t gini_cnt[MAX_NB_LEAF * 2 * MAX_CLASSES]
        DTYPE_t min_max[MAX_NB_LEAF * 2]

cdef class ClassificationCriterionDpu(Criterion):
    cdef SIZE_t * n_classes
    cdef SIZE_t sum_stride

cdef class GiniDpu(ClassificationCriterionDpu):
    cdef int dpu_update(self, CommandResults * res, SIZE_t eval_index, SIZE_t ndpu, SIZE_t n_node_samples,
                        SIZE_t * n_left, SIZE_t * n_right) nogil except -1
