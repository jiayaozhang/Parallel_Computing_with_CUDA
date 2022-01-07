#include <cstdio>

void cpu()
{
	printf("cpu\n");
}

/*
gpu function:
1. __global__
2. return type is void
*/
__global__
void gpu()
{
	printf("gpu\n");
}

int main()
{
	/*
	compile .cu:
	nvcc 00_HelloCUDA.cu -o 00_HelloCUDA
	*/
	cpu();
	/*
	call gpu function:
	1. <<<block, thread>>>
	2. cudaDeviceSynchronize();
	*/
	gpu<<<1, 1>>>();
	cudaDeviceSynchronize();
	return 0;
}