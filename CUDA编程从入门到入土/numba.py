from numba import cuda 
import numba
import numpy as np

MX = 20480
MY = 20480
TILE_DIM = 32
BLOCK_SIZE = 8


@cuda.jit 
def transpose(odata, idata):
    tile = cuda.shared.array((TILE_DIM,TILE_DIM), numba.types.float32)
    x = cuda.blockIdx.x * TILE_DIM + cuda.threadIdx.x 
    y = cuda.blockIdx.x * TILE_DIM + cuda.threadIdx.y 
    w = cuda.gridDim.x * TILE_DIM

    if x >= MX or y>= MY: return

    for i in range(0, TILE_DIM, BLOCK_SIZE):
        tile[cuda.threadIdx.y + i][cuda.threadIdx.x] = idata[y+i,x]

    cuda.syncthreads()
    x = cuda.blockIdx.y * TILE_DIM + cuda.threadIdx.x
    y = cuda.blockIdx.x * TILE_DIM + cuda.threadIdx.y

    for i in range(0, TILE_DIM, BLOCK_SIZE):
        odata[y+i,x] = tile[cuda.threadIdx.x, cuda.threadIdx.y + i]

threads = (TILE_DIM, BLOCK_SIZE)
blocks = ((MX + TILE_DIM - 1)) // TILE_DIM, (MY + TILE_DIM - 1) // TILE_DIM
a_in = cuda.to_device(np.arange(MX * MY, dtype = np.float32).reshape(MX,MY))
a_out = cuda.device_array_like(a_in)



# %timeit transpose[blocks, threads](a_in, a_out); cuda.synchronize()

