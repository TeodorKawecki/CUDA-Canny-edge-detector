#pragma once

#include<string>
#include<memory>
#include "IImage.h"

class IImgReader
{
public:
	virtual std::shared_ptr<IImage> ReadImage(const std::string &filePath) = 0;
};