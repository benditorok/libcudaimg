#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cstdlib>
#include "utils.cuh"

namespace utils
{
	inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort)
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
}
