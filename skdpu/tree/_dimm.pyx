# -*- coding: utf-8 -*-
"""DIMM memory manager module"""

# Authors: Sylvan Brocard
#
# License: MIT

import atexit

cdef extern from "src/trees.h":
    void free_dpus(dpu_set allset)

DEF CYTHON_DEBUG=0

_allocated = False  # whether the DPUs have been allocated
_requested_dpus = -1  # number of DPUs requested by user
_nr_dpus = -1  # number of DPUs currently allocated
_kernel = ""  # name of the currently loaded binary
_data_id = None  # ID of the currently loaded data

def free_all_dpus():
    global _allocated
    if _allocated:
        IF CYTHON_DEBUG >= 1:
            print("freeing dpus")
        free_dpus(allset)

atexit.register(free_all_dpus)
