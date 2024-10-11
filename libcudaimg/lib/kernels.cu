#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cmath>
#include "utils.cuh"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace
{
	__device__ const int SOBEL_X[3][3] = {
		{ -1, 0, 1 },
		{ -2, 0, 2 },
		{ -1, 0, 1 }
	};

	__device__ const int SOBEL_Y[3][3] = {
		{ -1, -2, -1 },
		{ 0, 0, 0 },
		{ 1, 2, 1 }
	};
}

namespace kernels
{
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

	__global__ void grayscaleImage(unsigned char* image, uint32_t width, uint32_t height)
	{
		uint32_t pixel_id = blockIdx.x * blockDim.x + threadIdx.x;
		uint32_t total_pixels = width * height;
		size_t rgb_index = pixel_id * 3;

		if (rgb_index >= total_pixels)
			return;

		unsigned char& r = image[rgb_index];
		unsigned char& g = image[rgb_index + 1];
		unsigned char& b = image[rgb_index + 2];

		// Calculate the grayscale value
		unsigned char gray = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);

		image[rgb_index] = gray;
		image[rgb_index + 1] = gray;
		image[rgb_index + 2] = gray;
	}

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

			uint32_t idx = threadIdx.x;

			if (idx < 256)
			{
				shared_cdf[idx] = 0;

				if (idx == 0)
				{
					// Initialize the first value of CDF
					shared_cdf[0] = (float)hist[0] / num_pixels;

					// Calculate the cumulative sum
					for (uint32_t i = 1; i < 256; ++i)
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
			uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
			uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

			if (x < width && y < height) {
				uint32_t idx = y * width + x; // Linear index from 2D coordinates
				output_img[idx] = (unsigned char)(255 * cdf[input_img[idx]]);
			}
		}
	}

	__global__ void boxFilter(unsigned char* image, unsigned char* output, uint32_t width, uint32_t height, uint32_t filter_size) {
		uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
		uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x < width && y < height) {
			int32_t half_filter = static_cast<int32_t>(filter_size) / 2;
			int32_t sum = 0;
			int32_t count = 0;

			// Loop over the filter window
			for (int32_t dy = -half_filter; dy <= half_filter; ++dy) {
				for (int32_t dx = -half_filter; dx <= half_filter; ++dx) {
					int32_t nx = min(max(static_cast<int32_t>(x) + dx, 0), static_cast<int32_t>(width) - 1);  // Clamp to image boundaries
					int32_t ny = min(max(static_cast<int32_t>(y) + dy, 0), static_cast<int32_t>(height) - 1); // Clamp to image boundaries
					sum += image[static_cast<uint32_t>(ny) * width + static_cast<uint32_t>(nx)];                 // Sum pixel values
					count++;
				}
			}

			// Compute the average and write to output image
			output[y * width + x] = sum / count;
		}
	}

	__global__ void gaussianBlur(unsigned char* image, unsigned char* output, uint32_t width, uint32_t height, float sigma) {
		uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
		uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

		// Calculate the filter radius based on sigma
		int32_t filter_radius = static_cast<int32_t>(ceil(3 * sigma));

		if (x < width && y < height) {
			float sum = 0.0f;
			float normalization_factor = 0.0f;

			// Loop over the filter window
			for (int32_t dy = -filter_radius; dy <= filter_radius; ++dy) {
				for (int32_t dx = -filter_radius; dx <= filter_radius; ++dx) {
					int32_t nx = min(max(static_cast<int32_t>(x) + dx, 0), static_cast<int32_t>(width) - 1);  // Clamp to image boundaries
					int32_t ny = min(max(static_cast<int32_t>(y) + dy, 0), static_cast<int32_t>(height) - 1); // Clamp to image boundaries

					// Calculate the Gaussian weight
					float distance = dx * dx + dy * dy;
					float weight = expf(-(distance) / (2.0f * sigma * sigma)) / (2.0f * M_PI * sigma * sigma);

					sum += image[static_cast<uint32_t>(ny) * width + static_cast<uint32_t>(nx)] * weight; // Sum weighted pixel values
					normalization_factor += weight;
				}
			}

			// Normalize the result and write to output image
			output[y * width + x] = static_cast<unsigned char>(sum / normalization_factor);
		}
	}

	__global__ void sobelEdgeDetection(const unsigned char* image, unsigned char* output, uint32_t width, uint32_t height)
	{
		uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
		uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x + 2 >= width || y >= height)
			return;

		float gradient_x = 0.0f;
		float gradient_y = 0.0f;

		// Apply the Sobel filter
		for (int32_t dy = -1; dy <= 1; ++dy) {
			for (int32_t dx = -1; dx <= 1; ++dx) {
				// Clamp coordinates to image boundaries
				int32_t nx = min(max(static_cast<int32_t>(x) + dx, 0), static_cast<int32_t>(width) - 1);
				int32_t ny = min(max(static_cast<int32_t>(y) + dy, 0), static_cast<int32_t>(height) - 1);

				// Fetch the pixel value from the grayscale image
				unsigned char pixel_value = image[ny * width + nx];

				// Apply Sobel filters to compute gradients
				gradient_x += pixel_value * SOBEL_X[dy + 1][dx + 1];
				gradient_y += pixel_value * SOBEL_Y[dy + 1][dx + 1];
			}
		}

		// Compute the gradient magnitude
		float magnitude = sqrtf(gradient_x * gradient_x + gradient_y * gradient_y);

		// Clamp the result to [0, 255] and write it to the output image
		output[y * width + x] = static_cast<unsigned char>(min(max(magnitude, 0.0f), 255.0f));
	}
}
