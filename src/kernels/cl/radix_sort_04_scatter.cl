#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#define MASK_WORDS (GROUP_SIZE / 32)

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* values,
    __global const uint* offsets,
    __global       uint* output,
    unsigned int n,
    unsigned int shift,
    unsigned int groups)
{
    const unsigned int lid = get_local_id(0);
    const unsigned int gid = get_global_id(0);
    const unsigned int group = get_group_id(0);

    __local uint masks[RADIX_SIZE * MASK_WORDS];
    for (unsigned int j = lid; j < RADIX_SIZE * MASK_WORDS; j += GROUP_SIZE)
        masks[j] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    const bool valid = gid < n;
    uint value = 0;
    uint digit = 0;
    if (valid) {
        value = values[gid];
        digit = (value >> shift) & (RADIX_SIZE - 1);
        atomic_or(&masks[digit * MASK_WORDS + (lid >> 5)], 1u << (lid & 31));
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (valid) {
        uint rank = 0;
        for (unsigned int w = 0; w < (lid >> 5); ++w)
            rank += popcount(masks[digit * MASK_WORDS + w]);
        rank += popcount(masks[digit * MASK_WORDS + (lid >> 5)] & ((1u << (lid & 31)) - 1u));
        output[offsets[digit * groups + group] + rank] = value;
    }
}
