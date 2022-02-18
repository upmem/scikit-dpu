import os
import subprocess

import numpy
from numpy.distutils.misc_util import Configuration

dpu_pkg_config_args = subprocess.getoutput("dpu-pkg-config --libs --cflags dpu").split()
dpu_pkg_config_libs = [arg[2:] for arg in dpu_pkg_config_args if arg.startswith("-l")]
dpu_pkg_config_lib_dirs = [arg[2:] for arg in dpu_pkg_config_args if arg.startswith("-L")]
dpu_pkg_config_include = [arg[2:] for arg in dpu_pkg_config_args if arg.startswith("-I")]

def configuration(parent_package="", top_path=None):
    config = Configuration("tree", parent_package, top_path)
    libraries = []
    if os.name == "posix":
        libraries.append("m")

    # choose one:
    # extra_compile_args = ["-g"]  # debug configuration
    extra_compile_args = ["-O3"]  # release configuration

    # compiling DPU binaries and adding them as a resource
    # print("Compiling DPU binary")
    # output = subprocess.run(
    #     ["dpu-upmem-dpurte-clang", "-DNR_TASKLETS=3", "-DSTACK_SIZE_DEFAULT=256", "-DSTACK_SIZE_TASKLET_1=2048",
    #      "-O2", "-o", "tasklet_stack_check.dpu", "tasklet_stack_check.c"], cwd='skdpu/tree/src/dpu_programs')
    # print(output)
    # config.add_data_files('src/dpu_programs/tasklet_stack_check.dpu')

    subprocess.check_output(
        ["dpu-upmem-dpurte-clang", "-DNR_TASKLETS=16", "-DSIZE_BATCH=32", "-O2",
         "-o", "trees_dpu_kernel_v2", "trees_dpu_kernel_v2.c"], cwd='skdpu/tree/src/dpu_programs')
    config.add_data_files('src/dpu_programs/trees_dpu_kernel_v2')

    # config.add_extension(
    #     "_splitter_dpu",
    #     sources=["_splitter.pyx"],
    #     include_dirs=[numpy.get_include(), dpu_pkg_config_include],
    #     libraries=libraries + [dpu_pkg_config_libs],
    #     extra_compile_args=["-O3"],
    # )
    # config.add_extension(
    #     "_dpu",
    #     sources=["_dpu.pyx", "src/_dpu_c.c"],
    #     include_dirs=[numpy.get_include(), dpu_pkg_config_include],
    #     libraries=libraries + [dpu_pkg_config_libs],
    #     extra_compile_args=extra_compile_args,
    #     define_macros=[("NB_CLUSTERS", "12")],
    # )
    config.add_extension(
        "_dimm",
        sources=["_dimm.pyx", "src/dpu_management_v2.c"],
        include_dirs=[numpy.get_include()] + dpu_pkg_config_include,
        libraries=libraries + dpu_pkg_config_libs,
        library_dirs=dpu_pkg_config_lib_dirs,
        extra_compile_args=extra_compile_args,
    )
    config.add_extension(
        "_tree",
        sources=["_tree.pyx", "src/dpu_management_v2.c"],
        include_dirs=[numpy.get_include()] + dpu_pkg_config_include,
        libraries=libraries + dpu_pkg_config_libs,
        library_dirs=dpu_pkg_config_lib_dirs,
        extra_compile_args=extra_compile_args,
    )
    config.add_extension(
        "_splitter",
        sources=["_splitter.pyx", "src/dpu_management_v2.c"],
        include_dirs=[numpy.get_include()] + dpu_pkg_config_include,
        libraries=libraries + dpu_pkg_config_libs,
        library_dirs=dpu_pkg_config_lib_dirs,
        extra_compile_args=extra_compile_args,
    )
    config.add_extension(
        "_criterion",
        sources=["_criterion.pyx", "src/dpu_management_v2.c"],
        include_dirs=[numpy.get_include()] + dpu_pkg_config_include,
        libraries=libraries + dpu_pkg_config_libs,
        library_dirs=dpu_pkg_config_lib_dirs,
        extra_compile_args=extra_compile_args,
    )
    config.add_extension(
        "_utils",
        sources=["_utils.pyx", "src/dpu_management_v2.c"],
        include_dirs=[numpy.get_include()] + dpu_pkg_config_include,
        libraries=libraries + dpu_pkg_config_libs,
        library_dirs=dpu_pkg_config_lib_dirs,
        extra_compile_args=extra_compile_args,
    )

    config.add_subpackage("tests")

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(**configuration().todict())
