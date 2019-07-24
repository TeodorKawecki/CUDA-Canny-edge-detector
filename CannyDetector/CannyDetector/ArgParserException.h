#pragma once

#include <exception>

class ArgParserException : public std::exception
{
public:
	ArgParserException(const char *msg)
	{
		_msg = "ArgParserException: " + std::string(msg);
	}

public:
	const char * what() const throw ()
	{
		return _msg.c_str();
	}

private:
	std::string _msg;
};
