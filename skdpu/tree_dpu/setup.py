import os
import subprocess

import numpy
from numpy.distutils.misc_util import Configuration

# dpu_pkg_config_args = subprocess.getoutput("dpu-pkg-config --libs --cflags dpu").split()
dpu_pkg_config_libs = subprocess.getoutput("dpu-pkg-config --libs dpu")[2:]
dpu_pkg_config_include = subprocess.getoutput("dpu-pkg-config --cflags dpu")[2:]

def configuration(parent_package="", top_path=None):
    config = Configuration("tree_dpu", parent_package, top_path)
    libraries = []
    if os.name == "posix":
        libraries.append("m")
    # config.add_extension(
    #     "_splitter_dpu",
    #     sources=["_splitter_dpu.pyx"],
    #     include_dirs=[numpy.get_include(), dpu_pkg_config_include],
    #     libraries=libraries + [dpu_pkg_config_libs],
    #     extra_compile_args=["-O3"],
    # )
    config.add_extension(
        "_dpu",
        sources=["_dpu.pyx", "src/_dpu_c.c"],
        # include_dirs=[numpy.get_include()],
        include_dirs=[numpy.get_include(), dpu_pkg_config_include],
        # libraries=libraries,
        libraries=libraries + [dpu_pkg_config_libs],
        # extra_compile_args=["-O3"] + dpu_pkg_config_args,
        extra_compile_args=["-O3"],
        define_macros=[("NB_CLUSTERS","12")],
    )

    # config.add_subpackage("tests")

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(**configuration().todict())
