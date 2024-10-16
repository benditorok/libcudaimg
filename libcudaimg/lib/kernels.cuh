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
	__global__ void sobelEdgeDetection(const unsigned char* image, unsigned char* output, uint32_t width, uint32_t height);

	// Apply a laplace edge detection to the image
	__global__ void laplaceEdgeDetection(const unsigned char* image, unsigned char* output, uint32_t width, uint32_t height);

	namespace harris
	{
		// Compute the gradients of the image
		__global__ void computeGradients(const unsigned char* input, float* grad_x, float* grad_y, uint32_t width, uint32_t height);

		// Compute the Harris response of the image
		__global__ void computeHarrisResponse(const float* grad_x, const float* grad_y, float* response, uint32_t width, uint32_t height, float k);

		// Apply non-maximum suppression to the Harris response
		__global__ void nonMaxSuppression(const float* response, unsigned char* output, uint32_t width, uint32_t height, float threshold);
	}
}
