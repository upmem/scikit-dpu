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
#          Nelson Liu <nelson@nelsonliu.me>
#
# License: BSD 3 clause

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

cdef class DpuTreeBuilder(TreeBuilder):
    cdef Params p
    cdef SIZE_t ndpu