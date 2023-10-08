#include "common_test.hpp"

int main() {

  facethink::facethinkcaffe::NetBuilder<float> net_builder;
  {
    std::cout<<"################# SSD Det test ################"<<std::endl;
    auto net = net_builder.Create("../data/ict_ssd/SSD_deploy20181022.prototxt",
                                  "../data/ict_ssd/SSD_deploy20181022.caffemodel");

    std::fstream fin("../data/ssd/det/pydata", std::ios::in);
    std::vector<float> fdata;
    float ff;
    while(fin >> ff){
      fdata.push_back(ff);
    }
    fin.close();
    std::vector<int> shape = { 1, 3, 300, 300 };

    auto blob = net->blob("data");
    blob->Reshape(shape);
    blob->ImportFromExtenalData(fdata.data(), 1*3*300*300);

    net->Forward();

    PrintBlob(net->blob("detection_out"), 100);
  }
}
