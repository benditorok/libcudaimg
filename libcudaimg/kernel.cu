#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);

        if (abort)
        {
            cudaDeviceReset();
            exit(code);
        }
    }
}

// Invert all the pixels in the image
__global__ void invertImage(unsigned char* image, uint32_t width, uint32_t height) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        uint32_t index = y * width + x;
        image[index] = 255 - image[index]; // Invert pixel value
    }
}

/// <summary>
/// Exposed function which calls the invertImage kernel.
/// </summary>
/// <param name="image">Pointer to the first byte of the image.</param>
/// <param name="image_len">The number of bytes in the image.</param>
/// <param name="width">The width of the image, should be multiplied by 3 if it's in an RGB format.</param>
/// <param name="height">The height of the image.</param>
extern "C" __declspec(dllexport)
void invertImage(unsigned char* image, uint32_t image_len, uint32_t width, uint32_t height) {
    unsigned char* d_image;
    size_t imageSize = image_len * sizeof(unsigned char);

    // Allocate memory on the GPU
    gpuErrchk(cudaMalloc((void**)&d_image, imageSize));

    // Copy the image to device memory
    gpuErrchk(cudaMemcpy(d_image, image, imageSize, cudaMemcpyHostToDevice));

    // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);

    // Launch the kernel
    invertImage<<<gridSize, blockSize>>>(d_image, width, height);
    gpuErrchk(cudaGetLastError()); // Check for kernel launch errors

    // Copy the processed image back to the host
    gpuErrchk(cudaMemcpy(image, d_image, imageSize, cudaMemcpyDeviceToHost));

    // Free the device memory
    gpuErrchk(cudaFree(d_image));
}
