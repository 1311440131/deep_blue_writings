#include <stdio.h>
#include <cuda.h>

#define N  10000000

__global__ void vector_add(float *a, float  *b, float *out, int n) {
    int index = threadIdx.x;
    int stride = blockDim.x
    for(int i=0; i<n; i++) {
        out[i] = a[i] + b[i];
    }
}   

int main() {
    // declaring memory in host
    float *h_a, *b, *out;
    
    // declaring memory in device
    float *d_a;
    
    // allocating memory on host
    h_a   = (float*)malloc(sizeof(float) * N);
    b     = (float*)malloc(sizeof(float) * N);
    out   = (float*)malloc(sizeof(float) * N);
   
   // initializing h_a and b
    for(int i=0; i<N; i++) {
        h_a[i] = 0.1f;
        b[i]   = 0.2f;
    } 
    
    // allocating memory in device and copying host data  to device
    cudaMalloc((void**)&d_a, sizeof(float) * N);
    cudaMemcpy(d_a, h_a, sizeof(float) * N, cudaMemcpyHostToDevice);    
    vector_add<<<1,256>>>(h_a, b, out, N);
    
    // deallocating memory from device
    cudaFree(d_a);

    // deallocating memory from host
    free(h_a);
    free(b);
    free(out);
    
    // returning success
    return 0;
}