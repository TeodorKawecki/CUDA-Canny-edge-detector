#include <memory>
#include <opencv2/opencv.hpp>
#include "IImgReader.h"
#include "IImage.h"

class OpenCVImageReader : public IImgReader
{
public:
	virtual std::shared_ptr<IImage> ReadImage(const std::string &filePath) override;
};


