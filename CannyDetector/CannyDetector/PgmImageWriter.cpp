#include "PgmImageWriter.h"
#include "ImageIOException.h"
#include <fstream>

void PgmImageWriter::WriteImage(const std::string &path, std::shared_ptr<IImage> imagePtr) const
{
	std::ofstream pgmImage;

	try
	{
		pgmImage.open(path, std::ios::trunc);
	}
	catch (const std::ios_base::failure &exception)
	{
		throw ImageIOException("Could not open file to write!");
	}

	pgmImage << "P2\n";

	auto imageWidth = imagePtr->GetWidth();
	auto imageHight = imagePtr->GetHight();

	PixelType maxPixelValue = 0;

	for (auto i = 0; i < imageWidth * imageHight; i++)
	{
		if ((*imagePtr)[i] > maxPixelValue)
			maxPixelValue = (*imagePtr)[i];
	}

	pgmImage << imageWidth << " " << imageHight << " " << maxPixelValue <<"\n";

	for (auto i = 0; i < imageWidth * imageHight; i++)
	{
		pgmImage << (*imagePtr)[i] << " ";
	}

	pgmImage.close();
}
