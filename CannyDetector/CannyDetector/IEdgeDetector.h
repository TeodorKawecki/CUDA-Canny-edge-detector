#pragma once

#include "IImage.h"
#include <memory>

class IEdgeDetector
{
public:
	virtual std::shared_ptr<IImage> DetectEdges(std::shared_ptr<IImage> image) = 0;
};
