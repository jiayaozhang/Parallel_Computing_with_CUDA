#include <cstdio>

void cpu(int *a, int N)
{
	for (int i = 0; i < N; ++i)
	{
        a[i] = i;
	}
}

__global__
void gpu(int *a, int N)
{
	int threadi = blockIdx.x * blockDim.x + threadIdx.x;
	int step = gridDim.x * blockDim.x;
	for (int i = threadi; i < N; i += step)
	{
        a[i] *= 2;
	}
}

bool check(int *a, int N)
{
	for (int i = 0; i < N; ++i)
	{
        if (a[i] != i * 2)
        {
            return false;
        }
	}
    return true;
}

int main()
{
    /*
    um(Unified Memory): cpu gpu both can uses
    0. int *a;
    1. cudaMallocManaged(&a, N * sizeof(int));
    2. cudaFree(a);
    */
	const int N = 128;
    int *a;
    cudaMallocManaged(&a, N * sizeof(int));

	cpu(a, N);

	int threads = 16;
	int blocks = 16;
	gpu<<<blocks, threads>>>(a, N);
	cudaDeviceSynchronize();

    if (check(a, N))
    {
        printf("true\n");
    }
    else
    {
        printf("false\n");
    }

    cudaFree(a);
	return 0;
}