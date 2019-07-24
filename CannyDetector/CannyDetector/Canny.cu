#include "Canny.cuh"

void AllocGpuMemory(size_t imageSizeBytes, size_t anglesTableSizeBytes)
{
	gpuErrchk(cudaMallocManaged(&inputImageDev, imageSizeBytes));
	gpuErrchk(cudaMallocManaged(&bluredImageDev, imageSizeBytes));
	gpuErrchk(cudaMallocManaged(&magnitudeImageDev, imageSizeBytes));
	gpuErrchk(cudaMallocManaged(&angleImageDev, anglesTableSizeBytes));
	gpuErrchk(cudaMallocManaged(&surpressedNonMaximumImageDev, imageSizeBytes));
	gpuErrchk(cudaMallocManaged(&doubleThresholdImageDev, imageSizeBytes));
	gpuErrchk(cudaMallocManaged(&hysteresisImageDev, imageSizeBytes));
}

void FreeGpuMemory()
{
	gpuErrchk(cudaFree(inputImageDev));
	gpuErrchk(cudaFree(bluredImageDev));
	gpuErrchk(cudaFree(magnitudeImageDev));
	gpuErrchk(cudaFree(angleImageDev));
	gpuErrchk(cudaFree(surpressedNonMaximumImageDev));
	gpuErrchk(cudaFree(doubleThresholdImageDev));
	gpuErrchk(cudaFree(hysteresisImageDev));
}

__device__ void FixIndexes(int &pixelX, int &pixelY, size_t hightImg, size_t widthImg)
{
	if (pixelX < 0)
		pixelX = 0;

	if (pixelY < 0)
		pixelY = 0;

	if (pixelX >= hightImg)
		pixelX = hightImg - 1;

	if (pixelY >= widthImg)
		pixelY = widthImg - 1;
}

__global__ void GaussianFilterDevice(const PixelsPtrType inputImage, const size_t widthImg, const size_t hightImg, const size_t kernelWidth, PixelsPtrType outputImage)
{
	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int mainPixelX = threadIndex / widthImg;
	int mainPixelY = threadIndex % widthImg;

	if (threadIndex >= widthImg * hightImg)
	{
		return;
	}

	auto halfOfSide = (int)((kernelWidth - 1) / 2);
	int pixelsSum = 0;

	for (int row = -halfOfSide; row <= halfOfSide; ++row)
	{
		for (int col = -halfOfSide; col <= halfOfSide; ++col)
		{
			int currentPixelX = mainPixelX + row;
			int currentPixelY = mainPixelY + col;

			FixIndexes(currentPixelX, currentPixelY, hightImg, widthImg);

			auto kerX = row + halfOfSide;
			auto kerY = col + halfOfSide;

			pixelsSum += (PixelType)(1.0 / 159 * inputImage[currentPixelX * widthImg + currentPixelY] * GaussianKernelConstDevice[kerX * kernelWidth + kerY]);
		}
	}

	outputImage[threadIndex] = pixelsSum;
}

__global__ void FindIntensityGradientDevice(const PixelsPtrType inputImageDev, const size_t widthImg, const size_t hightImg, const size_t kernelWidth, AnglesPtrType angleImageDev, PixelsPtrType magnitudeImageDev)
{
	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int mainPixelX = threadIndex / widthImg;
	int mainPixelY = threadIndex % widthImg;

	if (threadIndex >= widthImg * hightImg)
	{
		return;
	}

	int halfOfSide = (int)((kernelWidth - 1) / 2);

	int sumX = 0;
	int sumY = 0;

	for (int row = -halfOfSide; row <= halfOfSide; ++row)
	{
		for (int col = -halfOfSide; col <= halfOfSide; ++col)
		{
			int currentPixelX = mainPixelX + row;
			int currentPixelY = mainPixelY + col;

			FixIndexes(currentPixelX, currentPixelY, hightImg, widthImg);

			int kerX = row + halfOfSide;
			int kerY = col + halfOfSide;

			sumX += inputImageDev[currentPixelX * widthImg + currentPixelY] * SobelXKernelConstDevice[kerX * kernelWidth + kerY];
			sumY += inputImageDev[currentPixelX * widthImg + currentPixelY] * SobelYKernelConstDevice[kerX * kernelWidth + kerY];
		}
	}

	__syncthreads();
	magnitudeImageDev[threadIndex] = (PixelType)(sqrtf(sumX * sumX + sumY * sumY));

	if (sumX == 0)
	{
		double value = PI / 2;
		if (sumY < 0)
		{
			value *= -1;
		}

		angleImageDev[threadIndex] = value;
	}
	else
	{
		angleImageDev[threadIndex] = atan(1.0 * sumY / sumX);
	}
}

__global__ void SurpressNonMaximumDevice(const PixelsPtrType inputImageDev, const size_t widthImg, const size_t hightImg, AnglesPtrType angleImageDev, PixelsPtrType magnitudeImageDev, PixelsPtrType nonMaxSurpressedImageDev)
{
	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int mainPixelX = threadIndex / widthImg;
	int mainPixelY = threadIndex % widthImg;

	if (threadIndex >= widthImg * hightImg)
	{
		return;
	}

	if (mainPixelX <= 0 || 
		mainPixelX >= hightImg - 1 ||
		mainPixelY <= 0 ||
		mainPixelY >= widthImg - 1)
	{
		nonMaxSurpressedImageDev[threadIndex] = 0;
		return;
	}

	double tangent = angleImageDev[threadIndex];

	PixelType pixelValue = magnitudeImageDev[threadIndex];

	nonMaxSurpressedImageDev[threadIndex] = pixelValue;

	if (((-22.5 < tangent) && (tangent <= 22.5)) || ((157.5 < tangent) && (tangent <= -157.5)))
	{
		if (pixelValue < magnitudeImageDev[mainPixelX * widthImg + mainPixelY + 1] || pixelValue < magnitudeImageDev[mainPixelX * widthImg + mainPixelY - 1])
		{
			pixelValue = 0;
		}
	}

	if (((-112.5 < tangent) && (tangent <= -67.5)) || ((67.5 < tangent) && (tangent <= 112.5)))
	{
		if (pixelValue < magnitudeImageDev[(mainPixelX + 1) * widthImg + mainPixelY] || pixelValue < magnitudeImageDev[(mainPixelX + 1) * widthImg + mainPixelY])
		{
			pixelValue = 0;
		}
	}

	if (((-67.5 < tangent) && (tangent <= -22.5)) || ((112.5 < tangent) && (tangent <= 157.5)))
	{
		if (pixelValue < magnitudeImageDev[(mainPixelX - 1) * widthImg + mainPixelY + 1] || pixelValue < magnitudeImageDev[(mainPixelX + 1) * widthImg + mainPixelY - 1])
		{
			pixelValue = 0;
		}
	}

	if (((-157.5 < tangent) && (tangent <= -112.5)) || ((22.5 < tangent) && (tangent <= 67.5)))
	{
		if (pixelValue < magnitudeImageDev[(mainPixelX + 1) * widthImg + mainPixelY + 1] || pixelValue < magnitudeImageDev[(mainPixelX - 1) * widthImg + mainPixelY - 1])
		{
			pixelValue = 0;
		}
	}

	nonMaxSurpressedImageDev[threadIndex] = pixelValue;
}


__global__ void DoubleThresholdDevice(const PixelsPtrType inputImageDev, const size_t widthImg, const size_t hightImg, PixelType lowBand, PixelType hightBand, PixelsPtrType doubleThresholdImageDev)
{
	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadIndex >= widthImg * hightImg)	
	{
		return;
	}

	if (inputImageDev[threadIndex] > hightBand)
	{
		doubleThresholdImageDev[threadIndex] = STRONG_EDGE;
	}
	else if (inputImageDev[threadIndex] > lowBand)
	{
		doubleThresholdImageDev[threadIndex] = WEAK_EDGE;
	}
	else
	{
		doubleThresholdImageDev[threadIndex] = NO_EDGE;
	}
}

__global__ void HysteresisDevice(const PixelsPtrType inputImageDev, const size_t widthImg, const size_t hightImg, PixelsPtrType hysteresisImageDev)
{
	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadIndex >= widthImg * hightImg)
	{
		return;
	}

	int mainPixelX = threadIndex / widthImg;
	int mainPixelY = threadIndex % widthImg;

	bool foundStrongEdge = false;

	for (int i = -1; i <= 1 && !foundStrongEdge; i++)
	{
		for (int j = -1; j <= 1; j++)
		{
			int currentPixelX = mainPixelX + i;
			int currentPixelY = mainPixelY + j;

			if (currentPixelX < 0 || currentPixelY < 0)
			{
				continue;
			}

			if (currentPixelX >= hightImg || currentPixelY >= widthImg)
			{
				continue;
			}

			if (inputImageDev[currentPixelX * widthImg + currentPixelY] == STRONG_EDGE)
			{
				foundStrongEdge = true;
				break;
			}
		}
	}

	if (!foundStrongEdge)
	{
		hysteresisImageDev[threadIndex] = NO_EDGE;
	}
	else
	{
		hysteresisImageDev[threadIndex] = STRONG_EDGE;
	}
}

PixelsPtrType CannyDetect(const PixelsPtrType inputImage, const size_t widthImg, const size_t hightImg, const PixelType lowHistBand, const PixelType highHistBand)
{
	auto imageSizeBytes = sizeof(PixelType) * widthImg * hightImg;
	auto anglesTableSizeBytes = sizeof(AngleType) * widthImg * hightImg;
	auto blocksNumber = (widthImg * hightImg) / THREADS_NUMBER;
	auto threadsNumber = THREADS_NUMBER;

	gpuErrchk(cudaMemcpy(inputImageDev, inputImage, imageSizeBytes, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpyToSymbol(GaussianKernelConstDevice, GaussianKernel, GAUSSIAN_KERNEL_WIDTH * GAUSSIAN_KERNEL_WIDTH * sizeof(int)));
	gpuErrchk(cudaMemcpyToSymbol(SobelXKernelConstDevice, SobelXKernel, SOBEL_KERNEL_WIDTH * SOBEL_KERNEL_WIDTH * sizeof(int)));
	gpuErrchk(cudaMemcpyToSymbol(SobelYKernelConstDevice, SobelYKernel, SOBEL_KERNEL_WIDTH * SOBEL_KERNEL_WIDTH * sizeof(int)));

	GaussianFilterDevice << <blocksNumber, threadsNumber >> > (inputImageDev, widthImg, hightImg, GAUSSIAN_KERNEL_WIDTH, bluredImageDev);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	FindIntensityGradientDevice << <blocksNumber, threadsNumber >> > (bluredImageDev, widthImg, hightImg, SOBEL_KERNEL_WIDTH, angleImageDev, magnitudeImageDev);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	SurpressNonMaximumDevice << <blocksNumber, threadsNumber >> > (magnitudeImageDev, widthImg, hightImg, angleImageDev, magnitudeImageDev, surpressedNonMaximumImageDev);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	DoubleThresholdDevice<< <blocksNumber, threadsNumber >> > (surpressedNonMaximumImageDev, widthImg, hightImg, lowHistBand, highHistBand, doubleThresholdImageDev);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	HysteresisDevice << <blocksNumber, threadsNumber >> > (doubleThresholdImageDev, widthImg, hightImg, hysteresisImageDev);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	auto outputImage = (PixelsPtrType)malloc(sizeof(PixelType) * widthImg * hightImg);

	gpuErrchk(cudaMemcpy(outputImage, hysteresisImageDev, imageSizeBytes, cudaMemcpyDeviceToHost));

	return outputImage;
}
