# Authors: Sylvan Brocard
#
# License: MIT

cimport numpy as np

ctypedef np.npy_uint32 UINT32_t

from sklearn.tree._tree cimport DTYPE_t          # Type of X
from sklearn.tree._tree cimport DOUBLE_t         # Type of y, sample_weight
from sklearn.tree._tree cimport SIZE_t           # Type for indices and counters
from sklearn.tree._tree cimport INT32_t          # Signed 32 bit integer
from sklearn.tree._tree cimport UINT32_t         # Unsigned 32 bit integer

cdef extern from "src/_dpu_c.c":
   void print_n_clusters_c()
   void allocate(UINT32_t *ndpu)
   void tasklet_stack()

def print_n_clusters():
    print_n_clusters_c()

def test_allocate():
    cdef UINT32_t ndpu
    cdef UINT32_t *ndpu_pointer = &ndpu
    # allocate(ndpu_pointer)
    allocate(&ndpu)
    print(f"here too NDPU = {ndpu}")

def test_tasklet_stack():
    tasklet_stack()

cdef class Testclass:
    cpdef DOUBLE_t var(self):
        return 1.25