/*
 * CUDA test program
 */

// Headers
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>

// Main program
int main(int argc, char *argv[])
{
    // CUDA devices
    int num_devices;
    cudaGetDeviceCount(&num_devices);

    printf("Number of devices: %d\n", num_devices);

    for (int device_id = 0; device_id < num_devices; device_id++)
    {
        cudaDeviceProp properties;
        cudaGetDeviceProperties(&properties, device_id);
        
        printf("               Device Number: %d\n", device_id);
        printf("                 Device name: %s\n", properties.name);
        printf("     Memory Clock Rate (MHz): %.1f\n", properties.memoryClockRate / 1000.0);
        printf("     Memory Bus Width (bits): %d\n", properties.memoryBusWidth);
        printf("Peak Memory Bandwidth (GB/s): %.1f\n", 2.0 * properties.memoryClockRate * (properties.memoryBusWidth / 8.0) / 1.0e6);
        printf("\n");
  }

    // End of program
    return EXIT_SUCCESS;
}
