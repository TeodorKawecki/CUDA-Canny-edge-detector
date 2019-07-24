#pragma once

namespace
{
	using PixelsPtrType = int *;
	using PixelType = int;

	using AnglesPtrType = double *;
	using AngleType = double;
}

class IImage
{
public:
	virtual ~IImage() = default;

public:
	virtual size_t GetWidth() const = 0;
	virtual size_t GetHight() const = 0;
	virtual PixelsPtrType GetHandle() const = 0;
	virtual size_t GetImageSize() const = 0;

public:
	virtual PixelType operator [] (unsigned index) const = 0;
};
