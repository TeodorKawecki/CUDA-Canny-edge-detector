#pragma once

#include <cuda.h>
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include "IImage.h"
#include "CannyConstants.h"
#include <iostream>

using namespace CannyConstants;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

constexpr static int THREADS_NUMBER = 1024;

__device__ PixelsPtrType inputImageDev;
__device__ PixelsPtrType bluredImageDev;
__device__ PixelsPtrType magnitudeImageDev;
__device__ AnglesPtrType angleImageDev;
__device__ PixelsPtrType surpressedNonMaximumImageDev;
__device__ PixelsPtrType doubleThresholdImageDev;
__device__ PixelsPtrType hysteresisImageDev;

__constant__ int GaussianKernelConstDevice[GAUSSIAN_KERNEL_WIDTH * GAUSSIAN_KERNEL_WIDTH];
__constant__ int SobelXKernelConstDevice[SOBEL_KERNEL_WIDTH * SOBEL_KERNEL_WIDTH];
__constant__ int SobelYKernelConstDevice[SOBEL_KERNEL_WIDTH * SOBEL_KERNEL_WIDTH];

PixelsPtrType CannyDetect(const PixelsPtrType inputImage, const size_t widthImg, const size_t hightImg, const PixelType lowHistBand, const PixelType highHistBand);
void AllocGpuMemory(size_t imageSizeBytes, size_t anglesTableSizeBytes);
void FreeGpuMemory();

__global__ void GaussianFilterDevice(const PixelsPtrType inputImageDev, const size_t widthImg, const size_t hightImg, const size_t kernelWidth, PixelsPtrType bluredImageDev);
__global__ void FindIntensityGradientDevice(const PixelsPtrType inputImageDev, const size_t widthImg, const size_t hightImg, const size_t kernelWidth, AnglesPtrType angleImageDev, PixelsPtrType magnitudeImageDev);
__global__ void SurpressNonMaximumDevice(const PixelsPtrType inputImageDev, const size_t widthImg, const size_t hightImg, AnglesPtrType angleImageDev, PixelsPtrType magnitudeImageDev, PixelsPtrType nonMaxSurpressedImageDev);
__global__ void DoubleThresholdDevice(const PixelsPtrType inputImageDev, const size_t widthImg, const size_t hightImg, PixelType lowBand, PixelType hightBand, PixelsPtrType doubleThresholdImageDev);
__global__ void HysteresisDevice(const PixelsPtrType inputImageDev, const size_t widthImg, const size_t hightImg, PixelsPtrType hysteresisImageDev);
