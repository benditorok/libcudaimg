#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

// Error checking macro
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA Error: %s (error code: %d)\n",            \
                    cudaGetErrorString(err), err);                          \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Kernel that inverts a grayscale image
__global__ void invertImage(unsigned char* image, uint32_t width, uint32_t height) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        uint32_t index = y * width + x;
        image[index] = 255 - image[index]; // Invert pixel value
    }
}

// Exposed function that will be called from the host
extern "C" __declspec(dllexport)
void invertImage(unsigned char* image, uint32_t image_len, uint32_t width, uint32_t height) {
    unsigned char* d_image;
    size_t imageSize = image_len * sizeof(unsigned char);

    // Allocate memory on the GPU
    CUDA_CHECK(cudaMalloc((void**)&d_image, imageSize));

    // Copy the image to device memory
    CUDA_CHECK(cudaMemcpy(d_image, image, imageSize, cudaMemcpyHostToDevice));

    // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);

    // Launch the kernel
    invertImage<<<gridSize, blockSize>>>(d_image, width, height);
    CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors

    // Copy the processed image back to the host
    CUDA_CHECK(cudaMemcpy(image, d_image, imageSize, cudaMemcpyDeviceToHost));

    // Free the device memory
    CUDA_CHECK(cudaFree(d_image));
}
