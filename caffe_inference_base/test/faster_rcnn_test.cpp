#include "common_test.hpp"

int main() {

  facethink::facethinkcaffe::NetBuilder<float> net_builder;

  auto net = net_builder.Create("../data/fast_rcnn/test.prototxt",
				"../data/fast_rcnn/mirror_model.caffemodel");

  std::fstream fin("../data/fast_rcnn/data.txt", std::ios::in);
  std::vector<float> fdata;
  float ff;
  while(fin >> ff){
    fdata.push_back(ff);
  }
  fin.close();
  std::vector<int> shape = { 1, 3, 562, 1000 };

  auto blob = net->blob("data");
  blob->Reshape(shape);
  blob->ImportFromExtenalData(fdata.data(), 1*3*562*1000);

  auto blob_im_info = net->blob("im_info");
  blob_im_info->Reshape(std::vector<int>{ 1, 3 });
  float aa[3] = { 562, 1000, 0.520833313 };
  blob_im_info->ImportFromExtenalData(aa, 3);

  //  net->PrintBlobs();
  facethink::RPNConfig<float> rpn_config;
  net->Forward(rpn_config);

  //net->PrintBlobs();
  //net->PrintLayers();
  PrintBlob(net->blob("head_bbox_pred"), 100);

  return 0;
}
