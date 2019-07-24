#include "ArgParserWrapper.h"
#include "ArgParserException.h"

ArgParserWrapper::ArgParserWrapper()
{
	_argParser = parser
	{
		{
			 {
			   "inputRelativeFilePath", {"-i", "--inputRelativeFilePath"},
			   "Input relative file path.", 1
			 },
			 {
			   "outputRelativeFilePath", {"-o", "--outputRelativeFilePath"},
			   "Output relative file path.", 1
			 },
			 {
			   "cudaMode", {"-c","--cuda"},
			   "Cuda mode (default: off)", 0
			 },
			 {
			   "lowHistBand", {"-l", "--lowHistBand"},
			   "Lower hysteresis band.", 1
			 },
			 {
			   "highHistBand", {"-h", "--highHistBand"},
			   "Higher hysteresis band.", 1
			 },
			 {
			   "repeat", {"-r","--repeat"},
			   "Repeat", 1
			 },
			 {
			   "surpressWritingImage", {"-s","--surpressWritingImage"},
			   "Surpress writing image", 0
			 }
		}
	};
}

StartingParameters ArgParserWrapper::Parse(int argc, char ** argv)
{
	StartingParameters startingParameters;

	argagg::parser_results args;

	try
	{
		args = _argParser.parse(argc, argv);
	}
	catch (const std::exception& e)
	{
		throw ArgParserException(e.what());
	}

	if (args["inputRelativeFilePath"])
		startingParameters.inputRelativeFilePath = args["inputRelativeFilePath"].as<std::string>();

	if (args["outputRelativeFilePath"])
		startingParameters.outputRelativeFilePath = args["outputRelativeFilePath"].as<std::string>();

	if (args["lowHistBand"])
		startingParameters.lowHistBand = args["lowHistBand"].as<PixelType>();

	if (args["highHistBand"])
		startingParameters.highHistBand = args["highHistBand"].as<PixelType>();

	if (args["repeat"])
		startingParameters.repeatOperation = args["repeat"].as<size_t>();

	if (args["cudaMode"])
		startingParameters.cudaMode = true;

	if(args["surpressWritingImage"])
		startingParameters.surpressWritingImage = true;

	return startingParameters;
}
