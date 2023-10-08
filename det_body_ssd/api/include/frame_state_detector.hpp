#ifndef __FACETHINK_API_FRAME_STATE_DETECTOR_HPP__
#define __FACETHINK_API_FRAME_STATE_DETECTOR_HPP__

#include <string>
#include "config.hpp"
#include <caffe/caffe.hpp>
#include "det_body_ssd.hpp"
#include "CLock.h"
#ifdef WIN32
#include "windows.h"
#else
#endif
namespace facethink {
    namespace detbodyssd {

    class FrameStateDetector: public BodyDetectionSSD {
    public:
      explicit FrameStateDetector(
		  const std::string& det_model_prototxt,
		  const std::string& det_model_binary,
		  const std::string& config_file, 
		  const int gpu_id);

	  explicit FrameStateDetector(
		  const std::string& det_model_file,
		  const std::string& config_file, 
		  const int gpu_id);

	  virtual void detection(const cv::Mat& img, std::vector<cv::Rect>& rectangles, std::vector<float>& confidence);

    protected:
      cv::Mat DetNetPreprocessing(const cv::Mat& img, const cv::Size& size);
      cv::Mat ClsNetPreprocessing(const std::vector<cv::Mat>& imgs, const cv::Size& size);
      cv::Mat FilterResults(const cv::Mat& out, float threshold);
	  bool SetGPUID(const int gpu_id);
	  void LoadConfig(const std::string& config_file);

    private:

      std::shared_ptr<Net<float> > det_net_;
	  Config config_;
			// »¥³âËø  
	  static base::CLock::CCriticalSection m_criticalsection;
    };

}
}

#endif
