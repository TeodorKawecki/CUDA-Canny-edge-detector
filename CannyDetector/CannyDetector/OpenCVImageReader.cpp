#include "OpenCVImageReader.h"
#include "ImageIOException.h"
#include "Image.h"

std::shared_ptr<IImage> OpenCVImageReader::ReadImage(const std::string &filePath)
{
	cv::Mat image;

	image = cv::imread(filePath, cv::IMREAD_GRAYSCALE);

	if (!image.data)
	{
		throw ImageIOException("Could not open or find image!");
	}

	auto imgWidth = image.cols;
	auto imgHight = image.rows;

	auto pixelsGrayScale = (PixelsPtrType) malloc(sizeof(PixelType) * imgWidth * imgHight);

	for (auto row = 0; row < imgHight; ++row)
	{
		for (auto col= 0; col< imgWidth; ++col)
		{
			pixelsGrayScale[row * imgWidth + col] = (PixelType)image.at<uchar>(row, col);
		}
	}

	return std::make_shared<Image>(pixelsGrayScale, imgWidth, imgHight);
}