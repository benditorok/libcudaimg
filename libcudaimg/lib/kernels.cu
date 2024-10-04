#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cmath>

#include "utils.cuh"

namespace kernels
{
	// Invert all the pixels in the image
	__global__ void invertImage(unsigned char* image, uint32_t width, uint32_t height)
	{
		uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
		uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x < width && y < height)
		{
			uint32_t index = y * width + x;
			image[index] = 255 - image[index]; // Invert pixel value
		}
	}

	// Apply gamma transformation to the image
	__global__ void gammaTransformImage(unsigned char* image, uint32_t width, uint32_t height, float gamma)
	{
		uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
		uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x < width && y < height)
		{
			uint32_t index = y * width + x;
			image[index] = pow(image[index] / 255.0f, gamma) * 255; // Apply gamma transformation
		}
	}

	// Apply logarithmic transformation to the image
	__global__ void logarithmicTransformImage(unsigned char* image, uint32_t width, uint32_t height, float base)
	{
		uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
		uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x < width && y < height)
		{
			uint32_t index = y * width + x;
			image[index] = logf(1 + image[index]) / logf(1 + base) * 255; // Apply logarithmic transformation
		}
	}

	// Convert the image to grayscale
	__global__ void grayscaleImage(unsigned char* image, uint32_t width, uint32_t height)
	{
		uint32_t pixel_id = blockIdx.x * blockDim.x + threadIdx.x;
		uint32_t total_pixels = width * height;
		size_t rgb_index = pixel_id * 3;

		if (rgb_index < total_pixels)
		{
			unsigned char& r = image[rgb_index];
			unsigned char& g = image[rgb_index + 1];
			unsigned char& b = image[rgb_index + 2];

			// Calculate the grayscale value
			unsigned char gray = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);

			image[rgb_index] = gray;
			image[rgb_index + 1] = gray;
			image[rgb_index + 2] = gray;
		}
	}

	// Compute the histogram of the image
	__global__ void computeHistogram(unsigned char* image, uint32_t* histogram, uint32_t width, uint32_t height)
	{
		uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
		uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x < width && y < height) {
			uint32_t index = y * width + x;
			unsigned char& pixelValue = image[index];

			atomicAdd(&histogram[pixelValue], 1);
		}
	}

	namespace histogram_balancing
	{
		__global__ void computeCDF(const uint32_t* hist, float* cdf, uint32_t num_pixels)
		{
			__shared__ float shared_cdf[256];

			int idx = threadIdx.x;

			if (idx < 256) 
			{
				shared_cdf[idx] = 0;

				if (idx == 0) 
				{
					// Initialize the first value of CDF
					shared_cdf[0] = (float)hist[0] / num_pixels;

					// Calculate the cumulative sum
					for (int i = 1; i < 256; ++i) 
					{
						shared_cdf[i] = shared_cdf[i - 1] + (float)hist[i] / num_pixels;
					}
				}
			}

			__syncthreads();

			// Copy to global memory
			if (idx < 256) 
			{
				cdf[idx] = shared_cdf[idx];
			}
		}

		__global__ void applyEqualization(const unsigned char* input_img, unsigned char* output_img, const float* cdf, uint32_t width, uint32_t height) {
			int x = blockIdx.x * blockDim.x + threadIdx.x;
			int y = blockIdx.y * blockDim.y + threadIdx.y;

			if (x < width && y < height) {
				int idx = y * width + x; // Linear index from 2D coordinates
				output_img[idx] = (unsigned char)(255 * cdf[input_img[idx]]);
			}
		}
	}
}
