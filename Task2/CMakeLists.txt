cmake_minimum_required(VERSION 3.10)

project(MatrixMulCuda LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_STANDARD 11)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")

find_package(CUDA REQUIRED)
message(STATUS "CUDA Version: ${CUDA_VERSION}")

add_executable(matrix_mul_block matrix_mul_modified.cu)

target_link_libraries(matrix_mul_block ${CUDA_LIBRARIES})