#include <__clang_cuda_builtin_vars.h>
#include <stdio.h>

#define TILE_DIM 32
#define BLOCK_SIZE 8
#define mx 2048
#define my 2048

__global__ void transpose(float* odata, float* idata)
{
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int w = gridDim.x * TILE_DIM;
    if( x >= mx || y >= my) return;
    for (int i = 0 ; i < TILE_DIM; i += BLOCK_SIZE)
    {
        odata[x * w + y + i] = idata[(y+i)*w +x];
    }

}


__global__ void transpose2(float* odata, float* idata)
{
    __shared__ float tile[TILE_DIM][TILE_DIM];
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int w = gridDim.x * TILE_DIM;
    if( x >= mx || y >= my) return;
    for (int i = 0 ; i < TILE_DIM; i += BLOCK_SIZE)
    {
        tile[threadIdx.y + i][threadIdx.x] = idata[(y + i) * w + x];
    }
    __syncthreads();
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    for( int i = 0; i < TILE_DIM; i += BLOCK_SIZE){
        odata[(y + i) * w + x] = tile[threadIdx.x][threadIdx.y + i];
    }
}

bool check(float *c_cpu, float* c_gpu)
{
    for (int r = 0; r < mx; r++){
        for (int c = 0; c < my; c++){
            if(c_cpu[r * mx + c] != c_gpu[r * my + c]){
                return false;
            }
        }
    }
    return true;
}


int main() 
{
    size_t size = mx * my * sizeof(float);
    float *h_idata, *h_odata, *d_idata, *d_odata, *res;
    cudaMallocHost(&h_idata, size);
    cudaMallocHost(&h_odata, size);
    cudaMallocHost(&res, size);
    cudaMalloc(&d_idata, size);
    cudaMalloc(&d_odata, size);
    dim3 threads(TILE_DIM, BLOCK_SIZE, 1);
    dim3 blocks((mx+TILE_DIM-1) / TILE_DIM , (my+TILE_DIM-1) / TILE_DIM ,1 );
    for(int i = 0; i < mx; i++)
    {
        for(int j = 0; j < my; j++)
        {
            h_idata[i*my+j] = i * my + j;
            res[i * my + j] = j * my + i;
        }
    }

    cudaEvent_t startEvent, stopEvent;
    float ms;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaMemcpy(d_idata, h_idata, size, cudaMemcpyHostToDevice);
    cudaEventRecord(startEvent,0);
    for(int i = 0 ; i < 100; i++)
        transpose<<<blocks, threads>>>(d_odata, d_idata);
    cudaEventRecord(stopEvent,0);
    transpose<<<blocks, threads>>>(d_odata, d_idata);
    cudaMemcpy(h_odata, d_odata, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    check(res, h_odata) ? printf("ok") : printf("error");
    cudaFreeHost(h_idata);
    cudaFreeHost(h_odata);
    cudaFreeHost(res);
    cudaFree(d_idata);
    cudaFree(d_odata);
}