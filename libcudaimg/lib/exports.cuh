#pragma once

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

	/// <summary>
	/// Exposed function which calls the gammaTransformImage kernel.
	/// </summary>
	/// <param name="image">Pointer to the first byte of the image.</param>
	/// <param name="image_len">The number of bytes in the image.</param>
	/// <param name="width">The width of the image, should be multiplied by 3 if it's in an RGB format.</param>
	/// <param name="height">The height of the image.</param>
	/// <param name="gamma">The gamma value to apply to the image.</param>
	extern "C" __declspec(dllexport)
		void gammaTransformImage(unsigned char* image, uint32_t image_len, uint32_t width, uint32_t height, float gamma);

	/// <summary>
	/// Exposed function which calls the logarithmicTransformImage kernel.
	/// </summary>
	/// <param name="image">Pointer to the first byte of the image.</param>
	/// <param name="image_len">The number of bytes in the image.</param>
	/// <param name="width">The width of the image, should be multiplied by 3 if it's in an RGB format.</param>
	/// <param name="height">The height of the image.</param>
	/// <param name="base">The base of the logarithmic transformation.</param>
	extern "C" __declspec(dllexport)
		void logarithmicTransformImage(unsigned char* image, uint32_t image_len, uint32_t width, uint32_t height, float base);

	/// <summary>
	/// Exposed function which calls the grayscaleImage kernel.
	/// </summary>
	/// <param name="image">Pointer to the first byte of the image.</param>
	/// <param name="image_len">The number of bytes in the image.</param>
	/// <param name="width">The width of the image, should be multiplied by 3 if it's in an RGB format.</param>
	/// <param name="height">The height of the image.</param>
	extern "C" __declspec(dllexport)
		void grayscaleImage(unsigned char* image, uint32_t image_len, uint32_t width, uint32_t height);


	/// <summary>
	/// Exposed function which calls the computeHistogram kernel.
	/// </summary>
	/// <param name="image">Pointer to the first byte of the image.</param>
	/// <param name="image_len">The number of bytes in the image.</param>
	/// <param name="histogram">Pointer to the first uint32_t of the image. It should be 256 uint32_t long (0..=255).</param>
	/// <param name="width">The width of the image, should be multiplied by 3 if it's in an RGB format.</param>
	/// <param name="height">The height of the image.</param>
	extern "C" __declspec(dllexport)
		void computeHistogram(unsigned char* image, uint32_t image_len, uint32_t* histogram, uint32_t width, uint32_t height);

	/// <summary>
	/// Exposed function which calls the balanceHistogram kernel.
	/// </summary>
	/// <param name="image">Pointer to the first byte of the image.</param>
	/// <param name="image_len">The number of bytes in the image.</param>
	/// <param name="width">The width of the image, should be multiplied by 3 if it's in an RGB format.</param>
	/// <param name="height">The height of the image.</param>
	extern "C" __declspec(dllexport)
		void balanceHistogram(unsigned char* image, uint32_t image_len, uint32_t width, uint32_t height);

	/// <summary>
	/// Exposed function which calls the boxFilter kernel.
	/// </summary>
	/// <param name="image">Pointer to the first byte of the image.</param>
	/// <param name="image_len">The number of bytes in the image.</param>
	/// <param name="width">The width of the image, should be multiplied by 3 if it's in an RGB format.</param>
	/// <param name="height">The height of the image.</param>
	/// <param name="filterSize">The size of the filter to apply to the image.</param>
	void boxFilter(unsigned char* image, uint32_t image_len, uint32_t width, uint32_t height, uint32_t filterSize);
}
