#include <stdio.h>
#include <cuda.h>

#define N  10000000
#define BLOCK_SIZE  256

// __global__ indicates that function runs on GPU.
__global__ void vector_add(float *a, float  *b, float *out, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        out[i] = a[i] + b[i];
    }
}   


int main() {
    // declaring memory in host
    // h_a is a convention for host variable
    float *h_a, *b, *out;
    
    // declaring memory in device
    // d_a is the convention for device variable
    float *d_a;

    // allocating memory on host
    h_a   = (float*)malloc(sizeof(float) * N);
    b     = (float*)malloc(sizeof(float) * N);
    out   = (float*)malloc(sizeof(float) * N);
   
  // initializing a and v
    for(int i=0; i<N; i++) {
        h_a[i] = 0.1f;
        b[i]   = 0.2f;
    } 
    cudaMalloc((void**)&d_a, sizeof(float) * N);
    cudaMemcpy(d_a, h_a, sizeof(float) * N, cudaMemcpyHostToDevice);  
    
    int BLOCKS_NUM = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vector_add<<<BLOCKS_NUM, BLOCK_SIZE>>>(h_a, b, out, N);
    
    // deallocating memory from device
    cudaFree(d_a);

    // deallocating memory from host
    free(h_a);
    free(b);
    free(out);
  
    return 0;
}