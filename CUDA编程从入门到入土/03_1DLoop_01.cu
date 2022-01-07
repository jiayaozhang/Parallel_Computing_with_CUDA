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
	int threadi = blockIdx.x * blockDim.x + threadIdx.x;
	int step = gridDim.x * blockDim.x;
	for (int i = threadi; i < N; i += step)
	{
		printf("gpu, i: %d\n", i);
	}
}

int main()
{
	const int N = 128;
	cpu(N);

	int threads = 16;
	int blocks = 16;
	gpu<<<blocks, threads>>>(N);
	cudaDeviceSynchronize();
	return 0;
}