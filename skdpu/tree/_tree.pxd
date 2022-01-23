# Authors: Sylvan Brocard
#
# License: MIT

cimport numpy as np

from sklearn.tree._tree cimport DTYPE_t          # Type of X
from sklearn.tree._tree cimport DOUBLE_t         # Type of y, sample_weight
from sklearn.tree._tree cimport SIZE_t           # Type for indices and counters
from sklearn.tree._tree cimport INT32_t          # Signed 32 bit integer
from sklearn.tree._tree cimport UINT32_t         # Unsigned 32 bit integer

ctypedef np.npy_uint8 UINT8_t
ctypedef np.npy_uint16 UINT16_t
ctypedef np.npy_uint64 UINT64_t

from sklearn.tree._tree cimport Tree
from sklearn.tree._tree cimport TreeBuilder
from sklearn.tree._tree cimport Node
from sklearn.tree._splitter cimport Splitter

from sklearn.tree._splitter cimport SplitRecord

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

cdef class DpuTreeBuilder(TreeBuilder):
    cdef Params p