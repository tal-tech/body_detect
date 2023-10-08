#include "common_test.hpp"

int main(){
  const std::string prototxt = "../data/ssd_shuffle/ssd_face_detection_deploy.prototxt";
  const std::string model_file = "../data/ssd_shuffle/ssd_face_detection.caffemodel";

  facethink::facethinkcaffe::NetBuilder<float> net_builder;
  auto net = net_builder.Create(prototxt, model_file);

  std::fstream fin("../data/ssd_shuffle/pydata", std::ios::in);
  std::vector<float> fdata;
  float ff;
  while(fin >> ff){
    fdata.push_back(ff);
  }
  fin.close();

  std::vector<int> shape = { 1, 3, 46, 46 };
  auto blob_data = net->blob("data");
  blob_data->Reshape(shape);
  blob_data->ImportFromExtenalData(fdata.data(), 3*46*46);

  net->Forward();

  PrintBlob(net->blob("detection_out"), 100);

  return 0;
}
