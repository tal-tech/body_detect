#include "det_body_ssd.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <iomanip>
#include <vector>

using namespace facethink;


int main(int argc, char *argv[]) {
	
	if (argc < 4) {
		std::cerr << "Usage: " << argv[0]
			<< " det_model"
			<< " image path"
			<< " config file path" << std::endl;
		return 1;
	}

	const std::string det_model = argv[1];
	const std::string images_folder_path = argv[2];
	const std::string config_file_path = argv[3];
	
	BodyDetectionSSD *body_detector = BodyDetectionSSD::create(
		det_model,
		config_file_path);


	while (1) { //testing

		std::string file_list = images_folder_path + "list.txt";
		std::ifstream fin(file_list);
		if (!fin) {
			std::cout << "read file list failed" << std::endl;
			return -1;
		}

		//read images and annotations

		std::string file_string;
		while (std::getline(fin, file_string))
		{
			std::istringstream istr(file_string);
			std::string img_path, ano_path;
			istr >> img_path;
			istr >> ano_path;
			img_path = images_folder_path + img_path;
			ano_path = images_folder_path + ano_path;
			std::cout << img_path << std::endl;
			std::cout << ano_path << std::endl;

			cv::Mat img = cv::imread(img_path);
			if (img.data == 0) {
				std::cout << "read image failed:" << img_path << std::endl;
				continue;
			}

			auto time_start = std::chrono::steady_clock::now();

			std::vector<cv::Rect> rectangles;
			std::vector<float> confidences;
			body_detector->detection(img, rectangles, confidences);


			auto time_end = std::chrono::steady_clock::now();
		    std::cout<<"Cost Time: "<<std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count()<<" ms"<<std::endl;

			if (argc > 4) {
				for (int i = 0; i < rectangles.size(); i++)
				{
					int green = confidences[i] * 255;
					int red = (1 - confidences[i]) * 255;
					cv::rectangle(img, rectangles[i], cv::Scalar(0, green, red), 3);
				}

				double cost_time=std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
				cv::putText(img, std::to_string((int)cost_time) + "ms", cvPoint(3, 13), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 255, 0), 1, CV_AA);
				imshow("Face", img);

				if (cv::waitKey(0) == 27) {
					break;
				}
			}

		}

	} //while(1)
	
	return 0;
}
