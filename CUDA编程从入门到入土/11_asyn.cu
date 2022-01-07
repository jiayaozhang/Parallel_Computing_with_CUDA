#include "stdio.h"
#include "assert.h"
#define N 10000

__global__ void gpu(int *a, int *b, int *c_gpu){
    int r = blockDim.x * blockIdx.x + threadIdx.x;
    int c = blockDim.y * blockIdx.y + threadIdx.y;

    if( r < N && c < N)
    {
        c_gpu[r*N+c] = a[r*N+c] + b[r*N+c];
        a[r*N+c] = 1;
    }
}

void cpu(int *a, int *b, int *c_cpu){
    for(int r = 0; r < N; r++)
    {
        for(int c = 0; c < N ; c++){
            c_cpu[r * N + c] = a[r * N + c] + b[r * N + c] ;
        }
    }
}

bool check(int *c_cpu, int* c_gpu)
{
    for(int r = 0; r < N; r++)
    {
        for(int c = 0; c < N; c++)
        {
            if(c_cpu[r*N+c] != c_gpu[r*N+c])
            {
                return false;
            }
        }
    }
    return true;
}

inline cudaError_t checkCuda(cudaError_t result)
{
    if(result != cudaSuccess){
        fprintf(stderr, "CUDA runtime error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

int main() {
    int *a_cpu, *b_cpu, *a_gpu, *b_gpu, *c_cpu, *c_gpu, *c_gpu_cpu;
    size_t size = N * N * sizeof(int);

    cudaMallocHost(&a_cpu, size);
    cudaMallocHost(&b_cpu, size);
    cudaMallocHost(&c_cpu, size);
    cudaMallocHost(&c_gpu_cpu, size);
    cudaMalloc(&a_gpu, size);
    cudaMalloc(&b_gpu, size);
    cudaMalloc(&c_gpu, size);

    for( int r = 0 ; r < N; r++)
    {
        for(int c = 0; c < N; c++)
        {
            a_cpu[r*N+c]=r;
            b_cpu[r*N+c]=c;
            c_gpu_cpu[r*N+c] = 0;
            c_cpu[r*N+c] = 0;
        }
    }

    cpu(a_cpu, b_cpu, c_cpu);
    dim3 threads(16,16,1);
}