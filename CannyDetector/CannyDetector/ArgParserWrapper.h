#pragma once

#include "argagg.hpp"
#include "StartingParameters.h"
#include "IImage.h"

using argagg::parser_results;
using argagg::parser;

class ArgParserWrapper
{
public:
	ArgParserWrapper();

public:
	StartingParameters Parse(int argc, char **argv);

private:
	parser _argParser;
};
