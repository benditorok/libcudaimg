#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#include "utils.cuh"

namespace kernels
{
	// Invert all the pixels in the image
	__global__ void invertImage(unsigned char* image, uint32_t width, uint32_t height);

	// Apply gamma transformation to the image
	__global__ void gammaTransformImage(unsigned char* image, uint32_t width, uint32_t height, float gamma);

	// Apply logarithmic transformation to the image
	__global__ void logarithmicTransformImage(unsigned char* image, uint32_t width, uint32_t height, float base);

	// Turn the image into grayscale
	__global__ void grayscaleImage(unsigned char* image, uint32_t width, uint32_t height);

	// Compute the histogram of the image
	__global__ void computeHistogram(unsigned char* image, uint32_t* histogram, uint32_t width, uint32_t height);

	namespace histogram_balancing
	{
		// Compute the cumulative distribution function of the histogram
		__global__ void computeCDF(const uint32_t* hist, float* cdf, uint32_t num_pixels);

		// Apply histogram equalization to the image
		__global__ void applyEqualization(const unsigned char* input_img, unsigned char* output_img, const float* cdf, uint32_t width, uint32_t height);
	}

	// Apply a box filter to the image
	__global__ void boxFilter(unsigned char* image, unsigned char* output, uint32_t width, uint32_t height, uint32_t filterSize);

	// Apply a gaussian blur to the image
	__global__ void gaussianBlur(unsigned char* image, unsigned char* output, uint32_t width, uint32_t height, float sigma);

	// Apply a sobel edge detection to the image
	__global__ void sobelEdgeDetection(const unsigned char* input, unsigned char* output, uint32_t width, uint32_t height);
}
