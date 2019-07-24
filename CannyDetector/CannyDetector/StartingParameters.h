#pragma once

#include <string>

struct StartingParameters
{
	std::string inputRelativeFilePath = "";
	std::string outputRelativeFilePath = "default.pbm";
	size_t lowHistBand = 50;
	size_t highHistBand = 100;
	bool cudaMode = false;
	size_t repeatOperation = 1;
	bool surpressWritingImage = false;
};
