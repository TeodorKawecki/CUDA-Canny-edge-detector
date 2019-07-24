#include "CudaEdgeDetector.cuh"
#include "Image.h"

CudaEdgeDetector::CudaEdgeDetector(const PixelType lowHistBand, const PixelType highHistBand, const size_t widthImg, const size_t hightImg)
	: _lowHistBand(lowHistBand)
	, _highHistBand(highHistBand)
{
	auto imageSizeBytes = sizeof(PixelType) * widthImg * hightImg;
	auto anglesTableSizeBytes = sizeof(AngleType) * widthImg * hightImg;

	AllocGpuMemory(imageSizeBytes, anglesTableSizeBytes);
}

CudaEdgeDetector::~CudaEdgeDetector()
{
	FreeGpuMemory();
}

std::shared_ptr<IImage> CudaEdgeDetector::DetectEdges(std::shared_ptr<IImage> image)
{
	auto outputImage = CannyDetect(image->GetHandle(), image->GetWidth(), image->GetHight(), _lowHistBand, _highHistBand);

	return std::make_shared<Image>(outputImage, image->GetWidth(), image->GetHight());
}
