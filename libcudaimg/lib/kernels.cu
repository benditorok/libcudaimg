#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#include "utils.cuh"

namespace kernels
{
	// Invert all the pixels in the image
	__global__ void invertImage(unsigned char* image, uint32_t width, uint32_t height) {
		uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
		uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x < width && y < height) {
			uint32_t index = y * width + x;
			image[index] = 255 - image[index]; // Invert pixel value
		}
	}
}
