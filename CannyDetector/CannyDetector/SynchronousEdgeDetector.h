#pragma once

#include "IEdgeDetector.h"
#include "Image.h"
#include "CannyConstants.h"

using namespace CannyConstants;

class SynchronousEdgeDetector : public IEdgeDetector
{
public:
	explicit SynchronousEdgeDetector(const PixelType lowHistBand, const PixelType highHistBand, const size_t widthImg, const size_t hightImg);
	~SynchronousEdgeDetector();
	
public:
	virtual std::shared_ptr<IImage> DetectEdges(std::shared_ptr<IImage> image) override;

private:
	void GaussianFilter(const PixelsPtrType inputImage, const size_t widthImg, const size_t hightImg, const size_t kernelWidth, PixelsPtrType bluredImage);
	void FindIntensityGradient(const PixelsPtrType inputImage, const size_t widthImg, const size_t hightImg, const size_t kernelWidth, AnglesPtrType angleImageDev, PixelsPtrType magnitudeImage);
	void SurpressNonMaximum(const PixelsPtrType inputImage, const size_t widthImg, const size_t hightImg, AnglesPtrType angleImageDev, PixelsPtrType magnitudeImageDev, PixelsPtrType nonMaxSurpressedImage);
	void DoubleThreshold(const PixelsPtrType inputImage, const size_t widthImg, const size_t hightImg, PixelType lowBand, PixelType hightBand, PixelsPtrType doubleThresholdImage);
	void Hysteresis(const PixelsPtrType inputImage, const size_t widthImg, const size_t hightImg, PixelsPtrType hysteresisImage);
	void FixIndexes(int &pixelX, int &pixelY, size_t hightImg, size_t widthImg);

private:
	PixelType _lowHistBand;
	PixelType _highHistBand;
	size_t _imgWidth;
	size_t _imgHight;

private:
	PixelsPtrType inputImage;
	PixelsPtrType bluredImage;
	PixelsPtrType magnitudeImage;
	AnglesPtrType angleImage;
	PixelsPtrType surpressedNonMaximumImage;
	PixelsPtrType doubleThresholdImage;
	PixelsPtrType hysteresisImage;
};
