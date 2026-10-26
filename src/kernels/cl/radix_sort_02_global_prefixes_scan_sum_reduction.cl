#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* counts,
    __global       uint* sums,
    unsigned int level,      // номер порождаемого уровня (>=1)
    unsigned int src_size,   // число элементов на предыдущем уровне
    unsigned int src_offset, // смещение предыдущего уровня в sums (при level>1)
    unsigned int dst_offset) // смещение порождаемого уровня в sums
{
    const unsigned int i = get_global_id(0);
    const unsigned int dst_size = (src_size + 1) / 2;
    if (i >= dst_size)
        return;

    const unsigned int a = 2 * i;
    const unsigned int b = 2 * i + 1;
    uint sum;
    if (level == 1) {
        sum = counts[a] + (b < src_size ? counts[b] : 0);
    } else {
        sum = sums[src_offset + a] + (b < src_size ? sums[src_offset + b] : 0);
    }
    sums[dst_offset + i] = sum;
}
