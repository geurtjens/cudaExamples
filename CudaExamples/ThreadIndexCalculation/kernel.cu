
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void threadIndexCalculation()
{
    int gid = (blockIdx.x * blockDim.x) + threadIdx.x;
    printf("(%d * %d) + %d = %d\n", blockIdx.x, blockDim.x, threadIdx.x, gid);
}
int main()
{
    printf("(blockIdx.x * blockDim.x) + threadIdx.x = gid\n");

    threadIndexCalculation << <2, 4 >> > ();
}