#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#include "utils.cuh"
#include "exports.cuh"
#include "kernels.cuh"

using namespace utils;

namespace exports
{
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
		kernels::invertImage <<<gridSize,blockSize>>> (d_image, width, height);
		gpuErrchk(cudaGetLastError()); // Check for kernel launch errors

		// Copy the processed image back to the host
		gpuErrchk(cudaMemcpy(image, d_image, imageSize, cudaMemcpyDeviceToHost));

		// Free the device memory
		gpuErrchk(cudaFree(d_image));
	}

	void gammaTransformImage(unsigned char* image, uint32_t image_len, uint32_t width, uint32_t height) {
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
		kernels::gammaTransformImage << <gridSize, blockSize >> > (d_image, width, height);
		gpuErrchk(cudaGetLastError()); // Check for kernel launch errors

		// Copy the processed image back to the host
		gpuErrchk(cudaMemcpy(image, d_image, imageSize, cudaMemcpyDeviceToHost));

		// Free the device memory
		gpuErrchk(cudaFree(d_image));
	}
}
