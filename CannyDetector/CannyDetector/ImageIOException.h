#pragma once

#include <exception>

class ImageIOException : public std::exception
{
public:
	ImageIOException(const char *msg)
	{
		_msg = "ImageIOException: " + std::string(msg);
	} 

public:
	const char * what() const throw ()
	{
		return _msg.c_str();
	}

private:
	std::string _msg;
};
