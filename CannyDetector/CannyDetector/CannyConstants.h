#pragma once

#include "IImage.h"

namespace CannyConstants
{
	static constexpr PixelType STRONG_EDGE = 255;
	static constexpr PixelType WEAK_EDGE = 122;
	static constexpr PixelType NO_EDGE = 0;
	static constexpr double PI = 3.141592654f;

	static constexpr size_t GAUSSIAN_KERNEL_WIDTH = 5;
	static constexpr size_t SOBEL_KERNEL_WIDTH = 3;
	static constexpr int GaussianKernel[GAUSSIAN_KERNEL_WIDTH * GAUSSIAN_KERNEL_WIDTH] = { 2, 4, 5, 4, 2, 4, 9, 12, 9, 4, 5,  12,  15, 12, 5, 4, 9, 12, 9, 4, 2, 4, 5, 4, 2 };
	static constexpr int SobelXKernel[SOBEL_KERNEL_WIDTH * SOBEL_KERNEL_WIDTH] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	static constexpr int SobelYKernel[SOBEL_KERNEL_WIDTH * SOBEL_KERNEL_WIDTH] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
} // CannyConstants
