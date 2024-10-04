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
		kernels::invertImage<<<gridSize, blockSize>>>(d_image, width, height);
		gpuErrchk(cudaGetLastError()); // Check for kernel launch errors
		gpuErrchk(cudaDeviceSynchronize());

		// Copy the processed image back to the host
		gpuErrchk(cudaMemcpy(image, d_image, imageSize, cudaMemcpyDeviceToHost));

		// Free the device memory
		gpuErrchk(cudaFree(d_image));
	}

	void gammaTransformImage(unsigned char* image, uint32_t image_len, uint32_t width, uint32_t height, float gamma) {
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
		kernels::gammaTransformImage<<<gridSize, blockSize>>>(d_image, width, height, gamma);
		gpuErrchk(cudaGetLastError()); // Check for kernel launch errors
		gpuErrchk(cudaDeviceSynchronize());

		// Copy the processed image back to the host
		gpuErrchk(cudaMemcpy(image, d_image, imageSize, cudaMemcpyDeviceToHost));

		// Free the device memory
		gpuErrchk(cudaFree(d_image));
	}

	void logarithmicTransformImage(unsigned char* image, uint32_t image_len, uint32_t width, uint32_t height, float base)
	{
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
		kernels::logarithmicTransformImage << <gridSize, blockSize >> > (d_image, width, height, base);
		gpuErrchk(cudaGetLastError()); // Check for kernel launch errors
		gpuErrchk(cudaDeviceSynchronize());

		// Copy the processed image back to the host
		gpuErrchk(cudaMemcpy(image, d_image, imageSize, cudaMemcpyDeviceToHost));

		// Free the device memory
		gpuErrchk(cudaFree(d_image));
	}

	void grayscaleImage(unsigned char* image, uint32_t image_len, uint32_t width, uint32_t height)
	{
		unsigned char* d_image;
		size_t imageSize = image_len * sizeof(unsigned char);

		// Allocate memory on the GPU
		gpuErrchk(cudaMalloc((void**)&d_image, imageSize));

		// Copy the image to device memory
		gpuErrchk(cudaMemcpy(d_image, image, imageSize, cudaMemcpyHostToDevice));

		// Define block and grid sizes
		const uint32_t THREADS_PER_BLOCK = 256;
		uint32_t num_pixels = width * height;
		uint32_t blocks = (num_pixels + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

		// Launch the kernel
        kernels::grayscaleImage<<<blocks, THREADS_PER_BLOCK >>>(d_image, width, height);
		gpuErrchk(cudaGetLastError()); // Check for kernel launch errors
		gpuErrchk(cudaDeviceSynchronize());

		// Copy the processed image back to the host
		gpuErrchk(cudaMemcpy(image, d_image, imageSize, cudaMemcpyDeviceToHost));

		// Free the device memory
		gpuErrchk(cudaFree(d_image));
	}

	void computeHistogram(unsigned char* image, uint32_t image_len, uint32_t* histogram, uint32_t width, uint32_t height)
	{
		unsigned char* d_image;
		uint32_t* d_histogram;
		size_t imageSize = image_len * sizeof(unsigned char);
		size_t histogramSize = 256 * sizeof(uint32_t);

		// Allocate memory on the GPU
		gpuErrchk(cudaMalloc((void**)&d_image, imageSize));
		gpuErrchk(cudaMalloc((void**)&d_histogram, histogramSize));

		// Copy the image to device memory
		gpuErrchk(cudaMemcpy(d_image, image, imageSize, cudaMemcpyHostToDevice));

		// Define block and grid sizes
		const uint32_t THREADS_PER_BLOCK = 256;
		uint32_t num_pixels = width * height;
		uint32_t blocks = (num_pixels + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

		// Launch the kernel
		kernels::computeHistogram<<<blocks, THREADS_PER_BLOCK>>>(d_image, d_histogram, width, height);
		gpuErrchk(cudaGetLastError()); // Check for kernel launch errors
		gpuErrchk(cudaDeviceSynchronize());

		// Copy the histogram back to the host
		gpuErrchk(cudaMemcpy(histogram, d_histogram, histogramSize, cudaMemcpyDeviceToHost));

		// Free the device memory
		gpuErrchk(cudaFree(d_image));
		gpuErrchk(cudaFree(d_histogram));
	}

	//void balanceHistogram(unsigned char* image, uint32_t image_len, uint32_t width, uint32_t height)
	//{
	//	unsigned char* d_image;
	//	uint32_t* d_histogram;
	//	uint32_t* d_cdf;
	//	size_t imageSize = image_len * sizeof(unsigned char);
	//	size_t histogramSize = 256 * sizeof(uint32_t);

	//	// Allocate memory on the GPU
	//	gpuErrchk(cudaMalloc((void**)&d_image, imageSize));
	//	gpuErrchk(cudaMalloc((void**)&d_histogram, histogramSize));
	//	gpuErrchk(cudaMalloc((void**)&d_cdf, histogramSize));

	//	// Copy the image to device memory
	//	gpuErrchk(cudaMemcpy(d_image, image, imageSize, cudaMemcpyHostToDevice));
	//	gpuErrchk(cudaMemset(d_histogram, 0, histogramSize));

	//	// Define block and grid sizes
	//	dim3 blockSize(16, 16);
	//	dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);

	//	// Launch the kernel to compute the histogram
	//	kernels::computeHistogram << <gridSize, blockSize >> > (d_image, d_histogram, width, height);
	//	gpuErrchk(cudaGetLastError());
	//	gpuErrchk(cudaDeviceSynchronize());

	//	// Launch the kernel to compute the CDF
	//	kernels::histogram_utils::computeCDF << <1, 256 >> > (d_histogram, d_cdf, width * height);
	//	gpuErrchk(cudaGetLastError());
	//	gpuErrchk(cudaDeviceSynchronize());

	//	// Launch the kernel to normalize the image
	//	kernels::histogram_utils::normalizeImage << <gridSize, blockSize >> > (d_image, d_cdf, width, height, width * height);
	//	gpuErrchk(cudaGetLastError());
	//	gpuErrchk(cudaDeviceSynchronize());

	//	// Copy the processed image back to the host
	//	gpuErrchk(cudaMemcpy(image, d_image, imageSize, cudaMemcpyDeviceToHost));

	//	// Free the device memory
	//	gpuErrchk(cudaFree(d_image));
	//	gpuErrchk(cudaFree(d_histogram));
	//	gpuErrchk(cudaFree(d_cdf));
	//}

	void balanceHistogram(unsigned char* image, uint32_t image_len, uint32_t width, uint32_t height, unsigned char intensity)
	{
		unsigned char* d_image;
		uint32_t* d_histogram;
		uint32_t* d_cdf;
		size_t imageSize = image_len * sizeof(unsigned char);
		size_t histogramSize = 256 * sizeof(uint32_t);

		// Allocate memory on the GPU
		gpuErrchk(cudaMalloc((void**)&d_image, imageSize));
		gpuErrchk(cudaMalloc((void**)&d_histogram, histogramSize));
		gpuErrchk(cudaMalloc((void**)&d_cdf, histogramSize));

		// Copy the image to device memory
		gpuErrchk(cudaMemcpy(d_image, image, imageSize, cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemset(d_histogram, 0, histogramSize));

		// Define block and grid sizes
		dim3 blockSize(16, 16);
		dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);

		// Launch the kernel to compute the histogram
		kernels::computeHistogram << <gridSize, blockSize >> > (d_image, d_histogram, width, height);
		gpuErrchk(cudaGetLastError());
		gpuErrchk(cudaDeviceSynchronize());

		// Launch the kernel to compute the CDF
		kernels::histogram_utils::computeCDF << <1, 256 >> > (d_histogram, d_cdf, width * height);
		gpuErrchk(cudaGetLastError());
		gpuErrchk(cudaDeviceSynchronize());

		// Launch the kernel to normalize the image based on the provided intensity
		kernels::histogram_utils::normalizeImage << <gridSize, blockSize >> > (d_image, d_cdf, width, height, width * height, intensity);
		gpuErrchk(cudaGetLastError());
		gpuErrchk(cudaDeviceSynchronize());

		// Copy the processed image back to the host
		gpuErrchk(cudaMemcpy(image, d_image, imageSize, cudaMemcpyDeviceToHost));

		// Free the device memory
		gpuErrchk(cudaFree(d_image));
		gpuErrchk(cudaFree(d_histogram));
		gpuErrchk(cudaFree(d_cdf));
	}

	//void balanceHistogram(unsigned char* image, uint32_t image_len, uint32_t width, uint32_t height)
	//{
	//	unsigned char* d_image;
	//	size_t imageSize = image_len * sizeof(unsigned char);

	//	// Allocate memory on the GPU
	//	cudaMalloc((void**)&d_image, imageSize);

	//	// Copy the image to device memory
	//	cudaMemcpy(d_image, image, imageSize, cudaMemcpyHostToDevice);

	//	// Define block and grid sizes
	//	dim3 blockSize(16, 16);
	//	dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);

	//	// Launch the kernel to balance the histogram
	//	kernels::balanceHistogram << <gridSize, blockSize >> > (d_image, width, height);
	//	cudaDeviceSynchronize();

	//	// Copy the processed image back to the host
	//	cudaMemcpy(image, d_image, imageSize, cudaMemcpyDeviceToHost);

	//	// Free the device memory
	//	cudaFree(d_image);
	//}
}
