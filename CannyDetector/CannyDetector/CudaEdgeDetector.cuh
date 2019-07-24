#pragma once

#include <memory>
#include "IImage.h"
#include "Canny.cuh"
#include "IEdgeDetector.h"

class CudaEdgeDetector : public IEdgeDetector
{
public:
	explicit CudaEdgeDetector(const PixelType lowHistBand, const PixelType highHistBand, const size_t widthImg, const size_t hightImg);
	~CudaEdgeDetector();

public:
	virtual std::shared_ptr<IImage> DetectEdges(std::shared_ptr<IImage> image) override;

private:
	PixelType _lowHistBand;
	PixelType _highHistBand;
};

