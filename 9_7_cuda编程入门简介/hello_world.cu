#include <stdio.h>
#include <cuda.h>


__global__ void hello_world() {
    printf("Hello, World\n");
}

int main() {
    hello_world<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}