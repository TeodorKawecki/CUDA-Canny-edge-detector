#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <memory>
#include "IImgReader.h"
#include "OpenCVImageReader.h"
#include "PgmImageWriter.h"
#include "CudaEdgeDetector.cuh"
#include "SynchronousEdgeDetector.h"
#include "argagg.hpp"
#include "IOHelpers.h"
#include "ArgParserWrapper.h"
#include "StartingParameters.h"
#include "ArgParserException.h"
#include "DummyImage.h"
#include "ImageIOException.h"
#include <chrono>

int main(int argc, char **argv)
{
	StartingParameters startingParameters;
	ArgParserWrapper argParserWrapper;

	try
	{
		startingParameters = argParserWrapper.Parse(argc, argv);
	}
	catch (const ArgParserException &ex)
	{
		std::cerr << ex.what();
		return EXIT_FAILURE;
	}

	try
	{
		auto imageReader = std::make_unique<OpenCVImageReader>();
		auto imageWriter = std::make_unique<PgmImageWriter>();

		auto inputImage = imageReader->ReadImage(IOHelpers::GetExePath() + startingParameters.inputRelativeFilePath);

		std::unique_ptr<IEdgeDetector> detector = nullptr;

		if (startingParameters.cudaMode)
		{
			detector = std::make_unique<CudaEdgeDetector>(startingParameters.lowHistBand, startingParameters.highHistBand, inputImage->GetWidth(), inputImage->GetHight());
		}
		else
		{
			detector = std::make_unique<SynchronousEdgeDetector>(startingParameters.lowHistBand, startingParameters.highHistBand, inputImage->GetWidth(), inputImage->GetHight());
		}

		std::shared_ptr<IImage> outputImage = std::make_shared<DummyImage>();

		for (auto i = 0; i < startingParameters.repeatOperation; ++i)
		{
			auto begin = std::chrono::steady_clock::now();
			outputImage = detector->DetectEdges(inputImage);
			auto end = std::chrono::steady_clock::now();

			std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
		}

		if (!startingParameters.surpressWritingImage)
		{
			imageWriter->WriteImage(IOHelpers::GetExePath() + startingParameters.outputRelativeFilePath, outputImage);
		}
	}
	catch (const ImageIOException &e)
	{
		std::cerr << e.what();
		return EXIT_FAILURE;
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what();
		return EXIT_FAILURE;
	}
	
    return 0;
}
