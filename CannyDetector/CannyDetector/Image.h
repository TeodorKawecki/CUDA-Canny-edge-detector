#pragma once

#include "IImage.h"

class Image : public IImage
{
public:
	explicit Image(PixelsPtrType pixelsGrayScale, size_t imgWidth, size_t imgHight);
	~Image();

public:
	virtual size_t GetWidth() const override;
	virtual size_t GetHight() const override;
	virtual PixelsPtrType GetHandle() const override;
	virtual size_t GetImageSize() const override;

public:
	PixelType operator [] (unsigned index) const override;

private:
	size_t _imgWidth;
	size_t _imgHight;

private:
	PixelsPtrType _pixelsPtr;
};

