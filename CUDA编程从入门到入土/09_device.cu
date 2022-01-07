#include <cstdio>

int main()
{
    /*
    get device info:
    */
    int id;
    cudaGetDevice(&id);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, id);

    printf(
        "device, id: %d, sms: %d, capability major: %d, capability minor: %d, warp size: %d\n", 
        id, props.multiProcessorCount, props.major, props.minor, props.warpSize
    );
    return 0;
}