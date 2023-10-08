#include "common_test.hpp"

int main() {
  const std::string prototxt = "../data/landmark/landmark_deploy.prototxt";
  const std::string model_file = "../data/landmark/_iter_4000000.caffemodel";

  facethink::facethinkcaffe::NetBuilder<float> net_builder;
  auto net = net_builder.Create(prototxt, model_file);

  std::cout<<"################# Load Pycaffe data just for verification test ################"<<std::endl;
  std::fstream fin("../data/landmark/pydata", std::ios::in);
  std::vector<float> fdata;
  float ff;
  while(fin >> ff){
    fdata.push_back(ff);
  }
  fin.close();
  std::vector<int> shape = { 1, 1, 60, 60 };

  auto blob_data = net->blob("data");
  blob_data->Reshape(shape);
  blob_data->ImportFromExtenalData(fdata.data(), 60*60);

  net->Forward();

  PrintBlob(net->blob("Dense3"), 100);

  return 0;
}
