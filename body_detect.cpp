#include <iostream>
#include <string>
#include <vector>

#include "det_body_ssd.hpp"
#include "opencv2/core.hpp"
#include "json.h"

using namespace facethink;

BodyDetectionSSD *InitModel(const std::string &model, 
                            const std::string &config) {
    auto body_dect = BodyDetectionSSD::create(model, config);
    if (!body_dect) {
        std::cerr << "init model error" << std::endl;
        return nullptr;
    }
    return body_dect;
}

int DetectBody(BodyDetectionSSD *model, 
               const cv::Mat &img, 
               Json::Value &bodys) {
    std::vector<cv::Rect> rects;
    std::vector<float> confidences;
    model->detection(img, rects, confidences);

    if (rects.size() != confidences.size()) {
        std::cerr << "data abnormal" << std::endl;
        return -1;
    }
    for (auto i = 0; i != rects.size(); ++i) {
        Json::Value body;
        body["x"] = rects[i].x;
        body["y"] = rects[i].y;
        body["width"] = rects[i].width;
        body["height"] = rects[i].height;
        body["confidence"] = confidences[i];
        bodys.append(body);
    }

    return bodys.size();
}


int main(int argc, char *argv[]) {
    auto *model = InitModel("./ai_model/det_body_ssd_v1.0.3.bin", 
                            "./ai_model/config.ini");
    if (!model) {
        std::cerr << "InitModel error" << std::endl;
        return -1;
    }

    cv::Mat image = cv::imread("./body.jpeg");
    if (!image.data) {
        std::cerr << "read image error" << std::endl;
        return -1;
    }

    Json::Value bodys;
    auto count = DetectBody(model, image, bodys);
    if (count < 0) {
        std::cerr << "detect body error" << std::endl;
        return -1;
    }
    std::cout << "body count: " << count << std::endl;
    std::cout << bodys << std::endl;

    return 0;
}

