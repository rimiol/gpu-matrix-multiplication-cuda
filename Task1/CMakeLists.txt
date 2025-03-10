cmake_minimum_required(VERSION 3.10)

project(MatrixMulCuda LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_STANDARD 11)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")

find_package(CUDA REQUIRED)
message(STATUS "CUDA Version: ${CUDA_VERSION}")

add_executable(matrix_mul_original matrix_mul.cu)
add_executable(matrix_mul_optimized matrix_mul_optimized.cu)

target_link_libraries(matrix_mul_original ${CUDA_LIBRARIES})
target_link_libraries(matrix_mul_optimized ${CUDA_LIBRARIES})