#include "image_to_blob.hpp"

namespace facethink{

  cv::Mat blobFromImages(const std::vector<cv::Mat>& images, double scalefactor, bool swapRB){
    size_t i, nimages = images.size();
    if(nimages == 0)
      return cv::Mat();
    cv::Mat image0 = images[0];
    int nch = image0.channels();
    CV_Assert(image0.dims == 2);
    cv::Mat blob, image;
    if (nch == 3 || nch == 4){
      int sz[] = { (int)nimages, 3, image0.rows, image0.cols };
      blob = cv::Mat(4, sz, CV_32F);
      cv::Mat ch[4];

      for( i = 0; i < nimages; i++ ){
	cv::Mat image_ = images[i];
	if(image_.depth() == CV_8U){
	  image_.convertTo(image, CV_32F, scalefactor);
	}else{
	  image = image_;
	}
	CV_Assert(image.depth() == CV_32F);
	nch = image.channels();
	CV_Assert(image.dims == 2 && (nch == 3 || nch == 4));
	CV_Assert(image.size() == image0.size());

	for( int j = 0; j < 3; j++ )
	  ch[j] = cv::Mat(image.rows, image.cols, CV_32F, blob.ptr((int)i, j));
	if(swapRB)
	  std::swap(ch[0], ch[2]);
	cv::split(image, ch);
      }
    }else{
      CV_Assert(nch == 1);
      int sz[] = { (int)nimages, 1, image0.rows, image0.cols };
      blob = cv::Mat(4, sz, CV_32F);

      for( i = 0; i < nimages; i++ ){
	cv::Mat image_ = images[i];
	if(image_.depth() == CV_8U){
	  image_.convertTo(image, CV_32F, scalefactor);
	}else{
	  image = image_;
	}
	CV_Assert(image.depth() == CV_32F);
	nch = image.channels();
	CV_Assert(image.dims == 2 && (nch == 1));
	CV_Assert(image.size() == image0.size());

	image.copyTo(cv::Mat(image.rows, image.cols, CV_32F, blob.ptr((int)i, 0)));
      }
    }
    return blob;
  }

  cv::Mat blobFromImage(const cv::Mat& image_, double scalefactor, bool swapRB){
    std::vector<cv::Mat> images(1, image_);
    return blobFromImages(images, scalefactor, swapRB);
  }
}
