#include <cstdio>
#include <cassert>

int main()
{
	const int N = -1;
    int *a;
    cudaMallocManaged(&a, N * sizeof(int));

    /*
    cuda error:
    */
    cudaError_t err;
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("%s\n", cudaGetErrorString(err));
        assert(err == cudaSuccess);
    }

    cudaFree(a);
	return 0;
}