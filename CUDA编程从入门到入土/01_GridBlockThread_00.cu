#include <cstdio>

__global__
void gpu()
{
	/*
	gpu function value:
	* gridDim: number of block;
	* blockIdx: index of block;
	* blockDim: number of thread;
	* threadIdx: index of thread;
	*/
	printf(
        "gpu, gridDim.x: %d, blockIdx.x: %d, blockDim.x: %d, threadIdx.x: %d\n", 
        gridDim.x, blockIdx.x, blockDim.x, threadIdx.x
    );
}

int main()
{
	gpu<<<2, 4>>>();
	cudaDeviceSynchronize();
	return 0;
}