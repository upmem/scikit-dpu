for n_tasklets in $(seq 1 16)
do
    cd ../../../skdpu/tree/src/dpu_programs
    dpu-upmem-dpurte-clang -DNR_TASKLETS=$n_tasklets -DSIZE_BATCH=32 -O2 -o trees_dpu_kernel_v2 trees_dpu_kernel_v2.c
    cd -
    python tasklets.py $n_tasklets
done