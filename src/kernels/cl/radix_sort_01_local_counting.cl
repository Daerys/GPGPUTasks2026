#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* values,
    __global       uint* counts,
    unsigned int n,
    unsigned int shift,
    unsigned int groups)
{
    const unsigned int lid = get_local_id(0);
    const unsigned int gid = get_global_id(0);
    const unsigned int group = get_group_id(0);

    __local uint hist[RADIX_SIZE];
    if (lid < RADIX_SIZE)
        hist[lid] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gid < n) {
        const uint digit = (values[gid] >> shift) & (RADIX_SIZE - 1);
        atomic_inc(&hist[digit]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < RADIX_SIZE)
        counts[lid * groups + group] = hist[lid];
}
