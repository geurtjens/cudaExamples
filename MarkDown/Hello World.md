## CUDA Documentation

This can be found at:
https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

## Basic steps of a CUDA program
A cuda program will perform these steps multiple times.

1. Initialize data from CPU.
2. Transfer data from GPU to GPU
3. Launch Kernel using grid / block size.
4. Transfer results from GPU to CPU
5. Reclaim the allocated memory on both CPU and GPU

## Host vs Device
Host code runs on CPU whereas Device code runs on GPU

The host code calls Kernels on the Device.  Device code can also run device code via [CUDA Dynamic Parallelism](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-dynamic-parallelism).

CUDA has C++ language extensions called [Function Execution Space Specifiers](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#function-execution-space-specifiers)

So a kernel is specified as:

* `__global__` - function defining a `Kernel` must be executed with <<<>>>.
* `__device__` - function that runs on GPU device
* `__host__` - function that runs on CPU host.


## Kernels
This is how CPU calls GPU.  Its a function that runs asynchronously to the CPU and returns void.  Calling a kernel is referred to as `launching the kernel`.

``` c++
__global__ void helloWorld() {
    printf("hello world")
}
```
You return data from a kernel using parameters to the kernel.  Here we see we are passing two arrays and adding them up.

``` c++
__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
```

### threadIdx
The [threadIdx](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#threadidx) is implicitly initialised by cuda runtime from the launch parameters provided during kernel launch.  It is initialized based on the location of the thread in the thread block.

Lets change helloWord to include threadIdx

``` c++
__global__ void helloWorldFromThread() {
    printf("hello world from %d thread", threadIdx.x)
}
```
## Launching the Kernel
To launch a kernel you must specify `Kernel Launch Parameters`  

There are four possible parameters
* number_of_blocks
* threads_per_block
* shared memory size
* stream.

The minimum number of parameters that we will use here now are:
* number_of_blocks 
* threads_per_block.

Here we are passing 1,10 which means 1 block and 10 threads per block so that hello world will be printed 10 times.

We add [cudaDeviceSynchronize](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g10e20b05a95f638a4071a655503df25d) so that the CPU will wait until all previous launched kernels have finished their execution.
``` c++
int main() {
  hello_cuda<<<1,1>>>();
  cudaDeviceSynchronize();
  cudaDeviceReset()
  return 0;
}
```

The ways that the host can control the device are listed  in cuda runtime documentation [linked here](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html) including `cudaDeviceReset.`

## HelloWorld.cu
``` c++
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void helloWorldFromThread() {
  printf("Hello World, blockIdx.x: %d, threadIdx.x: %d\n", 
		  blockIdx.x, 
		  threadIdx.x);
}

int main()
{
  helloWorldFromThread<<<3, 4 >>>();
  cudaDeviceSynchronize();
  cudaDeviceReset();
  return 0;
}
```

This will return:
```
Hello World, blockIdx.x: 1, threadIdx.x: 0
Hello World, blockIdx.x: 1, threadIdx.x: 1
Hello World, blockIdx.x: 1, threadIdx.x: 2
Hello World, blockIdx.x: 1, threadIdx.x: 3
Hello World, blockIdx.x: 2, threadIdx.x: 0
Hello World, blockIdx.x: 2, threadIdx.x: 1
Hello World, blockIdx.x: 2, threadIdx.x: 2
Hello World, blockIdx.x: 2, threadIdx.x: 3
Hello World, blockIdx.x: 0, threadIdx.x: 0
Hello World, blockIdx.x: 0, threadIdx.x: 1
Hello World, blockIdx.x: 0, threadIdx.x: 2
Hello World, blockIdx.x: 0, threadIdx.x: 3
```
You can run it from the HelloWorld project in visual studio.

## Grids, Thread Blocks, Dim3
* Grid - a collection of all threads launched for a kernel
* Thread Block -  Threads in a grid are organized into groups called thread blocks.

## Dim3
Dim3 has 3 dimensions of x, y, z.  So you can specify for example how many threads there are in x separate for how many there are in y and how many in z.  This abstraction is useful considering GPU process graphics in up to 3 dimensions.  But we can also get away with using the one dimension of x.  We do this in the hello world example.  We are specifying only 1 dimension but putting the number 10 into the threads_per_block.

Dim3 is a uint3 data type.
``` c++
// ThreadsPerBlock
dim3 block(2,3,4)
printf("blockDim.x: %d\n", block.x);
printf("blockDim.y: %d\n", block.y);
printf("blockDim.z: %d\n", block.z);

// blocks per grid
dim3 grid(8, 4, 2)
printf("gridDim.x: %d\n", grid.x);
printf("gridDim.y: %d\n", grid.y);
printf("gridDim.z: %d\n", grid.z);
```

## Setting the number of blocks dynamically

``` c++
int main()
{
  // We want threads per block of X=8 and y=2
  dim threadsPerBlock(8,2);

  // The total word is x=16, y=4
  int total_x = 16;
  int total_y = 4;

  // So we work
  int bpg_x = total_x / threadsPerBlock.x;
  int bpg_y = total_y / threadsPerBlock.y;

  dim blocksPerGrid(bpg_x, bpg_y);
  
  run_kernel<<<threadsPerBlock,blocksPerGrid>>>();
}
```

## Limitations of block size

The max size of a block is 1024.

x * y * z < 1024

So x=32, y=16, z=16 is fine because 32 * 16 * 16 = 1024.

If you want an same number then best is 10 * 10 * 10 = 1000.

The x and y could be 1024 but the z <= 64.

Our blocks maximum is 2 to the power of 32 - 1.

Otherwise you get a Kernel Launch Failure

## blockDim

blockDim variable consists of the number of threads in each dimension of a thread block.  Notice all the thread block in a grid have the same block size, so this variable value is the same for all the threads in the grid.

## gridDim
The number of thread blocks in each dimension of a grid.

## Unique Index Calculation
Here is how we access elements of an array that was transferred to a kernel

When we have launched only on 1 thread block

``` c++
__global__ void unique_index_one_block(int * input) 
{
    int tid = threadIdx.x;
    printf("input[%d] = %d", tid, input[tid]);
}
```

Now lets launch a kernel with 2 blocks of 4 threads

``` c++
__global__ void threadIndexCalculation() 
{
  int gid = (blockIdx.x * blockDim.x) + threadIdx.x;
  printf("(%d * %d) + %d = %d", blockIdx.x, blockDim.x, threadIdx.x, gid);
}

int main()
{
  printf("(blockIdx.x * blockDim.x) + threadIdx.x = gid\n");
  threadIndexCalculation << <2, 4 >> > ();}
```

This `ThreadIndexCalculation` will produce.
```
(blockIdx.x * blockDim.x) + threadIdx.x = gid
(1 * 4) + 0 = 4
(1 * 4) + 1 = 5
(1 * 4) + 2 = 6
(1 * 4) + 3 = 7
(0 * 4) + 0 = 0
(0 * 4) + 1 = 1
(0 * 4) + 2 = 2
(0 * 4) + 3 = 3
```
And so all positions within the grid can be addressed.

## Memory Access Pattern
This depends on how we calculate our gid global index.

We prefer to calculate global indices in a way that threads within the same thread block access consecutive elements in an array.

block_offset = (number of threads in a block) * blockIdx.x

block_offset = (blockDim.x * blockDim.y) * blockIdx.x

