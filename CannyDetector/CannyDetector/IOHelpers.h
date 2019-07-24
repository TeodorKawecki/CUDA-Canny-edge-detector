#pragma once

#include <windows.h>
#include <iostream>

static class IOHelpers
{
public:
	static std::string GetExePath()
	{
		char buffer[MAX_PATH];
		GetModuleFileName(NULL, buffer, MAX_PATH);
		std::string::size_type pos = std::string(buffer).find_last_of("\\/");
		return std::string(buffer).substr(0, pos) + "\\";
	};
};
