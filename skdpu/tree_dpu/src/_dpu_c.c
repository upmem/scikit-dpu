#include <stdio.h>
#include <dpu.h>

#ifndef DPU_BINARY
#define DPU_BINARY "sklearn/tree_dpu/src/dpu_programs/tasklet_stack_check.dpu"
#endif

struct dpu_set_t allset;

static void print_n_clusters_c() {
    printf("%d\n",NB_CLUSTERS);
}

static void allocate(uint32_t *ndpu)
{
    DPU_ASSERT(dpu_alloc(DPU_ALLOCATE_ALL, NULL, &allset));
    DPU_ASSERT(dpu_get_nr_dpus(allset, ndpu));
    printf("ndpu = %d\n",*ndpu);
    DPU_ASSERT(dpu_free(allset));
}

static void tasklet_stack()
{
    struct dpu_set_t dpu; /* Iteration variable for the DPUs. */
    uint32_t each_dpu;    /* Iteration variable for the DPUs. */

    DPU_ASSERT(dpu_alloc(DPU_ALLOCATE_ALL, NULL, &allset));
    printf("allocated DPUs\n");

    DPU_ASSERT(dpu_load(allset, DPU_BINARY, NULL));
    printf("loaded binary\n");

    DPU_ASSERT(dpu_launch(allset, DPU_SYNCHRONOUS));
    printf("launched\n");

    DPU_FOREACH(allset, dpu, each_dpu) {
        if (each_dpu == 0)
            DPU_ASSERT(dpu_log_read(dpu, stdout));
    }

    DPU_ASSERT(dpu_free(allset));
}