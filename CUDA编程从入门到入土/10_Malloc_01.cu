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
    Page fault is low performance.
    If using um switched between cpu and gpu, system would raises page fault;
    We should use um after fetching memory manually to avoid page fault.
    */
	const int N = 128;
    int *a;
    cudaMallocManaged(&a, N * sizeof(int));

	cpu(a, N);

    /*
    from cpu to gpu
    */
    int id;
    cudaGetDevice(&id);
    cudaMemPrefetchAsync(a, N * sizeof(int), id);

	int threads = 16;
	int blocks = 16;
	gpu<<<blocks, threads>>>(a, N);
	cudaDeviceSynchronize();

    /*
    from gpu to cpu
    */
    cudaMemPrefetchAsync(a, N * sizeof(int), cudaCpuDeviceId);

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
