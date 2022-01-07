#include <cstdio>

void cpu_init(int *a, int N, int M)
{
	for (int row = 0; row < N; ++row)
	{
		for (int col = 0; col < M; ++col)
		{
			int i = row * M + col;
			a[i] = i;
		}
	}
}

__global__
void gpu_add(int *c, int *a, int *b, int N, int M)
{
	/*
	dim3
	*/
	int threadi_x = blockIdx.x * blockDim.x + threadIdx.x;
	int step_x = gridDim.x * blockDim.x;
	int threadi_y = blockIdx.y * blockDim.y + threadIdx.y;
	int step_y = gridDim.y * blockDim.y;
	for (int row = threadi_x; row < N; row += step_x)
	{
		for (int col = threadi_y; col < M; col += step_y)
		{
			int i = row * M + col;
			c[i] = a[i] + b[i];
		}
	}
}

bool check(int *a, int N, int M)
{
	for (int row = 0; row < N; ++row)
	{
		for (int col = 0; col < M; ++col)
		{
			int i = row * M + col;
			if (a[i] != i * 2)
			{
				return false;
			}
		}
	}
	return true;
}

int main()
{
	const int N = 128;
	const int M = 128;
    int *a, *b, *c;
    cudaMallocManaged(&a, N * M * sizeof(int));
    cudaMallocManaged(&b, N * M * sizeof(int));
    cudaMallocManaged(&c, N * M * sizeof(int));

	cpu_init(a, N, M);
	cpu_init(b, N, M);

	/*
	dim3
	*/
	dim3 threads(16, 16, 1);
	dim3 blocks(16, 16, 1);
	gpu_add<<<blocks, threads>>>(c, a, b, N, M);
	cudaDeviceSynchronize();

	if (check(c, N, M))
	{
		printf("true\n");
	}
	else
	{
		printf("false\n");
	}

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
	return 0;
}