#include "DummyImage.h"

size_t DummyImage::GetWidth() const
{
	return size_t();
}

size_t DummyImage::GetHight() const
{
	return size_t();
}

PixelType DummyImage::operator[](unsigned index) const
{
	return PixelType();
}

PixelsPtrType DummyImage::GetHandle() const
{
	return PixelsPtrType();
}

size_t DummyImage::GetImageSize() const
{
	return size_t();
}
