#pragma once

#include "IImgWriter.h"

class PgmImageWriter : public IImgWriter
{
public:
	PgmImageWriter() = default;
	~PgmImageWriter() = default;

public:
	virtual void WriteImage(const std::string &path, std::shared_ptr<IImage> imagePtr) const override;
};

