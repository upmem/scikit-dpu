# Authors: Sylvan Brocard
#
# License: MIT

cdef extern from "src/trees.h":
    ctypedef struct dpu_set:
        pass

cdef dpu_set allset  # set of all allocated DPUs
