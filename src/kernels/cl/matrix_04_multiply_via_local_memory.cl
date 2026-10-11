#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

#define TILE_SIZE GROUP_SIZE_X

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    const unsigned int lx = get_local_id(0);
    const unsigned int ly = get_local_id(1);
    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);

    __local float tile_a[TILE_SIZE][TILE_SIZE];
    __local float tile_b[TILE_SIZE][TILE_SIZE];

    float acc = 0.0f;

    const unsigned int tiles_count = (k + TILE_SIZE - 1) / TILE_SIZE;
    for (unsigned int t = 0; t < tiles_count; ++t) {
        const unsigned int a_col = t * TILE_SIZE + lx;
        const unsigned int b_row = t * TILE_SIZE + ly;
        tile_a[ly][lx] = (j < h && a_col < k) ? a[j * k + a_col] : 0.0f;
        tile_b[ly][lx] = (b_row < k && i < w) ? b[b_row * w + i] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int kk = 0; kk < TILE_SIZE; ++kk) {
            acc += tile_a[ly][kk] * tile_b[kk][lx];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (i < w && j < h) {
        c[j * w + i] = acc;
    }
}
