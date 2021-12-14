#include <dpu.h>

#define DPU_BINARY "trees_dpu_kernel_v2"

int main()
{
    struct dpu_set_t dpu_set, dpu;

    DPU_ASSERT(dpu_alloc(1, NULL, &dpu_set));
    DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));
    DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));

    DPU_FOREACH(dpu_set, dpu) {
        dpu_log_read(dpu, stdout);
    }

    return 0;
}
