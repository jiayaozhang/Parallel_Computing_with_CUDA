#include <cstdio>

void cpu(int N)
{
	for (int i = 0; i < N; ++i)
	{
		printf("cpu, i: %d\n", i);
	}
}

__global__
void gpu(int N)
{
	/*
	1D index: blockIdx.x * blockDim.x + threadIdx.x
	*/
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		printf("gpu, i: %d\n", i);
	}
}

int main()
{
	const int N = 128;
	cpu(N);

	int threads = 16;
	/*
	(N + a - 1) / a == ceil(N / a)
	*/
	int blocks = (N + threads - 1) / threads;
	gpu<<<blocks, threads>>>(N);
	cudaDeviceSynchronize();
	return 0;
}