#include "SynchronousEdgeDetector.h"

SynchronousEdgeDetector::SynchronousEdgeDetector(const PixelType lowHistBand, const PixelType highHistBand, const size_t widthImg, const size_t hightImg)
	: _lowHistBand(lowHistBand)
	, _highHistBand(highHistBand)
	, _imgWidth(widthImg)
	, _imgHight(hightImg)
{
	bluredImage = (PixelsPtrType)malloc(sizeof(PixelType) * _imgWidth * _imgHight);
	magnitudeImage = (PixelsPtrType)malloc(sizeof(PixelType) * _imgWidth * _imgHight);
	angleImage = (AnglesPtrType)malloc(sizeof(AngleType) * _imgWidth * _imgHight);
	surpressedNonMaximumImage = (PixelsPtrType)malloc(sizeof(PixelType) * _imgWidth * _imgHight);
	doubleThresholdImage = (PixelsPtrType)malloc(sizeof(PixelType) * _imgWidth * _imgHight);
	hysteresisImage = (PixelsPtrType)malloc(sizeof(PixelType) * _imgWidth * _imgHight);
}

SynchronousEdgeDetector::~SynchronousEdgeDetector()
{
	free(bluredImage);
	free(magnitudeImage);
	free(angleImage);
	free(surpressedNonMaximumImage);
	free(doubleThresholdImage);
}

std::shared_ptr<IImage> SynchronousEdgeDetector::DetectEdges(std::shared_ptr<IImage> image)
{
	GaussianFilter(image->GetHandle(), _imgWidth, _imgHight, GAUSSIAN_KERNEL_WIDTH, bluredImage);
	FindIntensityGradient(bluredImage, _imgWidth, _imgHight, SOBEL_KERNEL_WIDTH, angleImage, magnitudeImage);
	SurpressNonMaximum(magnitudeImage, _imgWidth, _imgHight, angleImage, magnitudeImage, surpressedNonMaximumImage);
	DoubleThreshold(surpressedNonMaximumImage, _imgWidth, _imgHight, _lowHistBand, _highHistBand, doubleThresholdImage);
	Hysteresis(doubleThresholdImage, _imgWidth, _imgHight, hysteresisImage);

	auto outputImage = (PixelsPtrType)malloc(sizeof(PixelType) * _imgWidth * _imgHight);
	memcpy(outputImage, hysteresisImage, sizeof(PixelType) * _imgWidth * _imgHight);

	return std::make_shared<Image>(outputImage, _imgWidth, _imgHight);
}

void SynchronousEdgeDetector::GaussianFilter(const PixelsPtrType inputImage, const size_t widthImg, const size_t hightImg, const size_t kernelWidth, PixelsPtrType outputImage)
{
	for (int mainPixelX = 0; mainPixelX < hightImg; ++mainPixelX)
	{
		for (int mainPixelY = 0; mainPixelY < widthImg; ++mainPixelY)
		{
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

					pixelsSum += (PixelType)(1.0 / 159 * inputImage[currentPixelX * widthImg + currentPixelY] * GaussianKernel[kerX * kernelWidth + kerY]);
				}
			}

			outputImage[mainPixelX * widthImg + mainPixelY] = pixelsSum;
		}
	}
}

void SynchronousEdgeDetector::FindIntensityGradient(const PixelsPtrType inputImageDev, const size_t widthImg, const size_t hightImg, const size_t kernelWidth, AnglesPtrType angleImageDev, PixelsPtrType magnitudeImageDev)
{
	for (int mainPixelX = 0; mainPixelX < hightImg; ++mainPixelX)
	{
		for (int mainPixelY = 0; mainPixelY < widthImg; ++mainPixelY)
		{
			auto oneDimIndex = mainPixelX * widthImg + mainPixelY;

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

					sumX += inputImageDev[currentPixelX * widthImg + currentPixelY] * SobelXKernel[kerX * kernelWidth + kerY];
					sumY += inputImageDev[currentPixelX * widthImg + currentPixelY] * SobelYKernel[kerX * kernelWidth + kerY];
				}
			}

			magnitudeImageDev[oneDimIndex] = (PixelType)(sqrtf(sumX * sumX + sumY * sumY));

			if (sumX == 0)
			{
				double value = PI / 2;
				if (sumY < 0)
					value *= -1;

				angleImageDev[oneDimIndex] = value;
			}
			else
				angleImageDev[oneDimIndex] = atan(1.0 * sumY / sumX);
		}
	}
}

void SynchronousEdgeDetector::SurpressNonMaximum(const PixelsPtrType inputImageDev, const size_t widthImg, const size_t hightImg, AnglesPtrType angleImageDev, PixelsPtrType magnitudeImageDev, PixelsPtrType nonMaxSurpressedImageDev)
{
	for (int mainPixelX = 0; mainPixelX < hightImg; ++mainPixelX)
	{
		for (int mainPixelY = 0; mainPixelY < widthImg; ++mainPixelY)
		{
			auto oneDimIndex = mainPixelX * widthImg + mainPixelY;

			if (mainPixelX <= 0 ||
				mainPixelX >= hightImg - 1 ||
				mainPixelY <= 0 ||
				mainPixelY >= widthImg - 1)
			{
				nonMaxSurpressedImageDev[oneDimIndex] = 0;
				continue;
			}

			double Tangent = angleImageDev[oneDimIndex];

			PixelType pixelValue = magnitudeImageDev[oneDimIndex];

			nonMaxSurpressedImageDev[mainPixelX * widthImg + mainPixelY] = pixelValue;

			if (((-22.5 < Tangent) && (Tangent <= 22.5)) || ((157.5 < Tangent) && (Tangent <= -157.5)))
			{
				if (pixelValue < magnitudeImageDev[mainPixelX * widthImg + mainPixelY + 1] || pixelValue < magnitudeImageDev[mainPixelX * widthImg + mainPixelY - 1])
				{
					pixelValue = 0;
				}
			}

			if (((-112.5 < Tangent) && (Tangent <= -67.5)) || ((67.5 < Tangent) && (Tangent <= 112.5)))
			{
				if (pixelValue < magnitudeImageDev[(mainPixelX + 1) * widthImg + mainPixelY] || pixelValue < magnitudeImageDev[(mainPixelX + 1) * widthImg + mainPixelY])
				{
					pixelValue = 0;
				}
			}

			if (((-67.5 < Tangent) && (Tangent <= -22.5)) || ((112.5 < Tangent) && (Tangent <= 157.5)))
			{
				if (pixelValue < magnitudeImageDev[(mainPixelX - 1) * widthImg + mainPixelY + 1] || pixelValue < magnitudeImageDev[(mainPixelX + 1) * widthImg + mainPixelY - 1])
				{
					pixelValue = 0;
				}
			}

			if (((-157.5 < Tangent) && (Tangent <= -112.5)) || ((22.5 < Tangent) && (Tangent <= 67.5)))
			{
				if (pixelValue < magnitudeImageDev[(mainPixelX + 1) * widthImg + mainPixelY + 1] || pixelValue < magnitudeImageDev[(mainPixelX - 1) * widthImg + mainPixelY - 1])
				{
					pixelValue = 0;
				}
			}

			nonMaxSurpressedImageDev[oneDimIndex] = pixelValue;
		}
	}
}


void SynchronousEdgeDetector::DoubleThreshold(const PixelsPtrType inputImageDev, const size_t widthImg, const size_t hightImg, PixelType lowBand, PixelType hightBand, PixelsPtrType doubleThresholdImageDev)
{
	for (int mainPixelX = 0; mainPixelX < hightImg; ++mainPixelX)
	{
		for (int mainPixelY = 0; mainPixelY < widthImg; ++mainPixelY)
		{
			int oneDimIndex = mainPixelX * widthImg + mainPixelY;

			if (oneDimIndex >= widthImg * hightImg)
			{
				continue;
			}

			if (inputImageDev[oneDimIndex] > hightBand)
				doubleThresholdImageDev[oneDimIndex] = STRONG_EDGE;
			else if (inputImageDev[oneDimIndex] > lowBand)
				doubleThresholdImageDev[oneDimIndex] = WEAK_EDGE;
			else
				doubleThresholdImageDev[oneDimIndex] = NO_EDGE;
		}
	}
}

void SynchronousEdgeDetector::Hysteresis(const PixelsPtrType inputImageDev, const size_t widthImg, const size_t hightImg, PixelsPtrType hysteresisImageDev)
{
	for (int mainPixelX = 0; mainPixelX < hightImg; ++mainPixelX)
	{
		for (int mainPixelY = 0; mainPixelY < widthImg; ++mainPixelY)
		{
			auto oneDimIndex = mainPixelX * widthImg + mainPixelY;

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
				hysteresisImageDev[oneDimIndex] = NO_EDGE;
			}
			else
			{
				hysteresisImageDev[oneDimIndex] = STRONG_EDGE;
			}
		}
	}
}

void SynchronousEdgeDetector::FixIndexes(int &pixelX, int &pixelY, size_t hightImg, size_t widthImg)
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