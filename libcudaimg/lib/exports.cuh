#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#include "utils.cuh"

namespace exports 
{
	/// <summary>
	/// Exposed function which calls the invertImage kernel.
	/// </summary>
	/// <param name="image">Pointer to the first byte of the image.</param>
	/// <param name="image_len">The number of bytes in the image.</param>
	/// <param name="width">The width of the image, should be multiplied by 3 if it's in an RGB format.</param>
	/// <param name="height">The height of the image.</param>
	extern "C" __declspec(dllexport)
		void invertImage(unsigned char* image, uint32_t image_len, uint32_t width, uint32_t height);
}
