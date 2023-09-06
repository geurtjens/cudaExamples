#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void helloWorldFromThread() {
	printf("Hello World, blockIdx.x: %d, threadIdx.x: %d\n", blockIdx.x,threadIdx.x);
}

int main()
{
	helloWorldFromThread << <3, 4 >> > ();
	cudaDeviceSynchronize();
	cudaDeviceReset();
	return 0;
}