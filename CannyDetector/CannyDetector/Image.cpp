#include "Image.h"

Image::Image(PixelsPtrType pixelsPtr, size_t imgWidth, size_t imgHight)
	: _pixelsPtr(pixelsPtr)
	, _imgWidth(imgWidth)
	, _imgHight(imgHight)
{

}

Image::~Image()
{
	if(_pixelsPtr != nullptr)
		delete _pixelsPtr;
}

size_t Image::GetWidth() const
{
	return _imgWidth;
}

size_t Image::GetHight() const
{
	return _imgHight;
}

PixelType Image::operator [] (unsigned index) const
{
	return _pixelsPtr[index];
}

PixelsPtrType Image::GetHandle() const
{
	return _pixelsPtr;
}

size_t Image::GetImageSize() const
{
	return sizeof(char) * _imgWidth * _imgHight;
}
