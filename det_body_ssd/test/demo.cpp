#include "det_body_ssd.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>

using namespace facethink;

int main(int argc, char *argv[]){

	if (argc < 4) {
		std::cerr << "Usage: " << argv[0]
			<< " det_model"
			<< " image path"
			<< " config file path" << std::endl;
		return 1;
	}

	const std::string det_model = argv[1];
	const std::string image_path = argv[2];
	const std::string config_file_path = argv[3];

	cv::Mat img = cv::imread(image_path); //test image
	if (img.data == 0) {
		std::cout << "read image failed:" << argv[7] << std::endl;
		return -1;
	}


  BodyDetectionSSD *body_detector = BodyDetectionSSD::create(
	  det_model,
	  config_file_path);

  std::vector<cv::Rect> rectangles;
  std::vector<float> confidences;
  body_detector->detection(img, rectangles, confidences);

  for (int i = 0; i < rectangles.size(); i++)
  {
	  std::cout << "id: " << i << std::endl;
	  std::cout << "confidence: " << confidences[i] << std::endl;
	  std::cout << "rectangle: " << rectangles[i] << std::endl;
  }
  std::getchar();
  return 0;
}
