#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    __global const uint* counts,
    __global const uint* sums,
    __global       uint* offsets,
    unsigned int m,
    unsigned int levels) // число уровней пирамиды, включая исходный массив
{
    const unsigned int i = get_global_id(0);
    if (i >= m)
        return;

    const unsigned int len = i + 1;
    const uint own = counts[i];
    uint total = (len & 1) ? own : 0; // уровень 0

    unsigned int level_size = m;
    unsigned int level_offset = 0;
    for (unsigned int k = 1; k < levels; ++k) {
        level_size = (level_size + 1) / 2;
        if ((len >> k) & 1)
            total += sums[level_offset + (len >> k) - 1];
        level_offset += level_size;
    }
    offsets[i] = total - own;
}
