#ifndef __FACETHINK_API_IMAGE_TO_BLOB_HPP__
#define __FACETHINK_API_IMAGE_TO_BLOB_HPP__

#include <opencv2/opencv.hpp>

namespace facethink{

  cv::Mat blobFromImages(const std::vector<cv::Mat>& images, double scalefactor, bool swapRB);
  cv::Mat blobFromImage(const cv::Mat& image_, double scalefactor, bool swapRB);
  
}

#endif
