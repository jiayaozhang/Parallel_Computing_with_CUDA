#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <iostream>

__global__ void add(float * a)
{
	a[threadIdx.x] = 1;
}

struct cudaDeviceProp{
	char name[256];
	size_t totalGlobalMem;
	size_t sharedMemPerBlock;
	int regsPerBlock;

}

int main(int argc, char **argv)
{
	int gpuCount = -1;
	cudaGetDeviceCount(&gpuCount);
	printf("gpuCount: %d \n", gpuCount);
	
	if(gpuCount < 0)
	{
		printf("no device! \n");
	}

	gpuCount++;
	cudaSetDevice(gpuCount-1);

	int deviceId;
	cudaGetDevice(&deviceId);
	printf("deviceId: %d\n", deviceId);

	float *aGpu;
	cudaMalloc((void**))&aGpu, 16 * sizeof(float));
	float a[16] = {0};
	cudaMemcpy(aGpu, a, 16 * sizeof(float), cudaMemcpyHostToDevice);
	kernelFunc<<<1,16>> >(aGpu);
	cudaMemcpy(a, aGpu, 16 * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 16 ; i++)
	{
		printf("%f", a[i]);
	}
	cudaFree(aGpu);
	cudaDeviceReset();

	int gpuCount = -1;
	cudaDeviceCount(&gpuCount);
	printf("gpuCount: %d\n", gpuCount);


	cudaGetDeviceProperties(&prop, 0);
	// printf("maxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
	// printf("maxThreadsDim: %d\n", prop.maxThreadsDim[0]);
	// printf("maxGridSize: %d\n", prop.maxGridSize[0]);
	// printf("totalConstMem: %d\n", prop.totalConstMem);
	// printf("clockRate: %d \n", prop.clockRate);
	// printf("integrated: %d \n", prop.integrated);
	printf("totalGlobalMem: %d\n", prop.totalGlobalMem);
}


// #include <stdio.h>
// #include <stdlib.h>
// #include <cuda_runtime_api.h>
// #include <iostream>

// __global__ void kernelFunc(float* a)
// {
//     a[threadIdx.x] = 1 ;
// }

// int main(int argc, char **argv)
// {
//     int gpuCount = -1;
//     cudaGetDeviceCount(&gpuCount);

//     printf("gpuCount: %d\n", gpuCount);

//     if (gpuCount <0)
//     {
//         printf("no device\n");
//     }

//     cudaSetDevice(gpuCount-1);
//     float *aGpu;
//     cudaMalloc((void**)&aGpu, 16 * sizeof(float));
//     float a[16] = {0};
//     cudaMemcpy(aGpu, a, 16*sizeof(float), cudaMemcpyHostToDevice);
//     kernelFunc<<<1, 16>>>(aGpu);
//     cudaMemcpy(a, aGpu, 16*sizeof(float), cudaMemcpyDeviceToHost);

//     for(int i=0; i<16; i++)
//     {
//         printf("%f \n", a[i]);
//     }

//     cudaFree(aGpu);
//     cudaDeviceReset();

// }
