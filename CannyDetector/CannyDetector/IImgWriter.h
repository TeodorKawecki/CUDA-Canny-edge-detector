#pragma once

#include<string>
#include "IImage.h"

class IImgWriter
{
public:
	virtual void WriteImage(const std::string &path, std::shared_ptr<IImage> imagePtr) const = 0;
};
