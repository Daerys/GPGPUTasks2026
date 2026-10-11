#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

#define TILE_SIZE GROUP_SIZE_X

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    const unsigned int lx = get_local_id(0); // меняется быстрее всего внутри warp
    const unsigned int ly = get_local_id(1);

    const unsigned int tile_i0 = get_group_id(0) * TILE_SIZE;
    const unsigned int tile_j0 = get_group_id(1) * TILE_SIZE;

    __local float tile[TILE_SIZE][TILE_SIZE + 1];

    {
        const unsigned int i = tile_i0 + lx;
        const unsigned int j = tile_j0 + ly;
        tile[ly][lx] = (i < w && j < h) ? matrix[j * w + i] : 0.0f;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    {
        const unsigned int out_row = tile_i0 + ly;
        const unsigned int out_col = tile_j0 + lx;
        if (out_row < w && out_col < h) {
            transposed_matrix[out_row * h + out_col] = tile[lx][ly];
        }
    }
}
