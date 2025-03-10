#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

#define N 1024

// #define BLOCK_SIZE 8
// #define BLOCK_SIZE 16
// #define BLOCK_SIZE 32
 #define BLOCK_SIZE 64

// Ядро для блочного умножения матриц
__global__ void matrixMulBlock(float *A, float *B, float *C, int width) {

    __shared__ float sharedA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sharedB[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int i = 0; i < width / BLOCK_SIZE; ++i) {
        // Загружаем фрагменты A и B в shared memory
        sharedA[threadIdx.y][threadIdx.x] = A[row * width + (i * BLOCK_SIZE + threadIdx.x)];
        sharedB[threadIdx.y][threadIdx.x] = B[(i * BLOCK_SIZE + threadIdx.y) * width + col];

        __syncthreads();  

        for (int j = 0; j < BLOCK_SIZE; ++j) {
            sum += sharedA[threadIdx.y][j] * sharedB[j][threadIdx.x];
        }

        __syncthreads();  
    }

    
    if (row < width && col < width) {
        C[row * width + col] = sum;
    }
}

void matrixMul(float *A, float *B, float *C, int width) {
    float *d_A, *d_B, *d_C;
    size_t size = width * width * sizeof(float);

    
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((width + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                   (width + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    
    matrixMulBlock<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, width);

   
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    int width = N;
    float *A = (float*)malloc(width * width * sizeof(float));
    float *B = (float*)malloc(width * width * sizeof(float));
    float *C = (float*)malloc(width * width * sizeof(float));

    
    for (int i = 0; i < width * width; ++i) {
        A[i] = rand() % 10;
        B[i] = rand() % 10;
    }

    
    clock_t start = clock();
    matrixMul(A, B, C, width);
    clock_t end = clock();
    
    
    printf("Time taken: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    free(A);
    free(B);
    free(C);

    return 0;
}
