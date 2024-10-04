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

	// Balance the histogram of the image
	__global__ void balanceHistogram(unsigned char* image, uint32_t width, uint32_t height);

	namespace histogram_utils
	{
		// Compute the cumulative distribution function of the histogram
		__global__ void computeCDF(uint32_t* histogram, uint32_t* cdf, uint32_t numPixels);

		// Normalize the image using the cumulative distribution function
		__global__ void normalizeImage(unsigned char* image, uint32_t* cdf, uint32_t width, uint32_t height, uint32_t numPixels, unsigned char intensity);

		//__global__ void normalizeImage(unsigned char* image, uint32_t* cdf, uint32_t width, uint32_t height, uint32_t numPixels);
	}
}
