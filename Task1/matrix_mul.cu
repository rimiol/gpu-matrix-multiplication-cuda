#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

#define N 2048  //размерность матриц

__global__ void matrixMul(float *A, float *B, float *C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width) {
        float sum = 0.0f;
        for (int i = 0; i < width; ++i) {
            sum += A[row * width + i] * B[i * width + col];
        }
        C[row * width + col] = sum;
    }
}

void matrixMulCUDA(float *A, float *B, float *C, int width) {
    float *d_A, *d_B, *d_C;
    size_t size = width * width * sizeof(float);

    // Выделение памяти на GPU
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Копирование данных в GPU
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Определение сетки и блоков
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (width + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    // Запуск ядра
    matrixMul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, width);

    
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

    // Заполняем A и B случайными значениями
    for (int i = 0; i < width * width; ++i) {
        A[i] = rand() % 10;
        B[i] = rand() % 10;
    }

    
    clock_t start = clock();
    matrixMulCUDA(A, B, C, width);
    clock_t end = clock();
    
   
    printf("Time taken: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    
    free(A);
    free(B);
    free(C);

    return 0;
}
