#include "frame_state_detector.hpp"
#include "image_to_blob.hpp"
#include<chrono>
#ifdef WIN32
#include <io.h>
#endif
using namespace base::CLock;

namespace facethink {
	namespace detbodyssd {
		base::CLock::CCriticalSection FrameStateDetector::m_criticalsection;
		FrameStateDetector::FrameStateDetector(
				const std::string& det_model_prototxt,
				const std::string& det_model_binary,
				const std::string& config_file,
				const int gpu_id) {

			LoadConfig(config_file);

			/*
			   if (!SetGPUID(gpu_id)) {
			   return;
			   }
			 */

			facethinkcaffe::NetBuilder<float> net_builder;
			det_net_ = net_builder.Create(det_model_prototxt, det_model_binary);

		}

		FrameStateDetector::FrameStateDetector(
				const std::string& det_model_file,
				const std::string& config_file,
				const int gpu_id) {

			LoadConfig(config_file);

			if (!SetGPUID(gpu_id)) {
				return;
			}

			facethinkcaffe::NetBuilder<float> net_builder;
			det_net_ = net_builder.Create(det_model_file);
		}

		bool FrameStateDetector::SetGPUID(const int gpu_id) {

			/*
#ifndef CPU_ONLY 
			//Caffe::DeviceQuery();
			bool id_flag = Caffe::CheckDevice(gpu_id);
			if (id_flag) {
			Caffe::SetDevice(gpu_id);
			BOOST_LOG_TRIVIAL(info) << "set GPU_ID: " << gpu_id << " succeed!";
			return true;
			}
			else {
			BOOST_LOG_TRIVIAL(error) << "set GPU_ID: " << gpu_id << " failed!";
			return false;
			}
#endif
			 */

			return true;
		}

		void FrameStateDetector::LoadConfig(const std::string& config_file) {

#ifdef WIN32
			if (_access(config_file.c_str(), 0) != -1) {
				config_.ReadIniFile(config_file);
			}
#else
			if (access(config_file.c_str(), 0) != -1) {
				config_.ReadIniFile(config_file);
			}
#endif
		}

		cv::Mat FrameStateDetector::DetNetPreprocessing(const cv::Mat& img,
				const cv::Size& size) {
			cv::Mat img_resize;
			cv::resize(img, img_resize, size);
			cv::Mat img_resize_float;
			img_resize.convertTo(img_resize_float, CV_32F);
			img_resize_float -= cv::Scalar(104, 117, 123);
			return blobFromImage(img_resize_float, 1.0f, false);
		}

		cv::Mat FrameStateDetector::FilterResults(const cv::Mat& out, float threshold) {
			if (fabs(out.at<float>(0,0) + 1) < 1e-5) { return cv::Mat(); }

			std::vector<int> rows_selected;
			for (int i=0; i<out.rows; ++i){
				if (out.at<float>(i, 2) > threshold) {
					rows_selected.push_back(i);
				}
			}

			cv::Mat new_out(rows_selected.size(), out.cols, out.type());
			for (size_t i=0; i<rows_selected.size(); ++i){
				out.row(rows_selected[i]).copyTo(new_out.row(i));
			}

			return new_out;
		}


		void FrameStateDetector::detection(const cv::Mat& img, std::vector<cv::Rect>& rectangles, std::vector<float>& confidence){

			Lock lock(&m_criticalsection);


			if (img.empty() || img.channels() != 3){
				BOOST_LOG_TRIVIAL(fatal)<<"Input image must has 3 channels.";
			}

			BOOST_LOG_TRIVIAL(debug) << "BodyDetNet: start.";

			auto time_start = std::chrono::steady_clock::now();

			auto det_input = det_net_->blob("data");
			std::vector<int> det_input_shape = det_input->shape();
			det_input_shape[0] = 1;
			det_input->Reshape(det_input_shape);

			cv::Mat det_blob_img = DetNetPreprocessing(img,
					cv::Size(det_input_shape[2], det_input_shape[3]));

			det_input->ImportFromExtenalData(det_blob_img.ptr<float>(0), det_input->count());
			auto ti = std::chrono::steady_clock::now();
			det_net_->Forward();
			std::cout << "Forward Cost Time: " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - ti).count() << " us" << std::endl;

			auto det_output = det_net_->blob("detection_out");
			const cv::Mat mat_det_out(det_output->shape(0),
					det_output->shape(1),
					CV_32FC1,
					const_cast<float *>(det_output->cpu_data()));

			cv::Mat mat_det_out_valid = FilterResults(mat_det_out, config_.BODY_BOX_THRESHOLD);

			if (!mat_det_out_valid.empty() && mat_det_out_valid.cols >=7)
			{
				for (size_t i = 0; i < mat_det_out_valid.rows; i++)
				{
					float conf = mat_det_out_valid.at<float>(i, 2);
					confidence.push_back(conf);

					float x1 = mat_det_out_valid.at<float>(i, 3);
					float y1 = mat_det_out_valid.at<float>(i, 4);
					float x2 = mat_det_out_valid.at<float>(i, 5);
					float y2 = mat_det_out_valid.at<float>(i, 6);

					int x1_ = static_cast<int>(x1 * img.cols);
					int y1_ = static_cast<int>(y1 * img.rows);
					int x2_ = static_cast<int>(x2 * img.cols);
					int y2_ = static_cast<int>(y2 * img.rows);

					cv::Rect rect(cv::Point2i(x1_, y1_), cv::Point2i(x2_, y2_));
					rectangles.push_back(rect);

				}
			}
			auto time_end = std::chrono::steady_clock::now();
			std::cout << "Body Det Net Cost Time: " << std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count() << " us" << std::endl;
		}

	}
}
