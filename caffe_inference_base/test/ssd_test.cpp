#include "common_test.hpp"

int main() {


  facethink::facethinkcaffe::NetBuilder<float> net_builder;
  {
    std::cout<<"################# SSD Det test ################"<<std::endl;
    auto net = net_builder.Create("../data/ssd/det/deploy.prototxt",
				  "../data/ssd/det/VGG_SSD_share_pose5_test_iter_72000.caffemodel");

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

  {
    std::cout<<"################# SSD Cls test ################"<<std::endl;
    const std::string prototxt = "../data/ssd/cls/deploy.prototxt";
    const std::string model_file = "../data/ssd/cls/body_iter_100000.caffemodel";

    auto net = net_builder.Create(prototxt, model_file);

    std::fstream fin("../data/ssd/cls/cls_data.txt", std::ios::in);
    std::vector<float> fdata;
    float ff;
    while(fin >> ff){
      fdata.push_back(ff);
    }
    fin.close();
    int n_det = fdata.size()/(3*128*128);

    std::vector<int> input_shape = { n_det, 3, 128, 128 };

    auto blob = net->blob("data");
    blob->Reshape(input_shape);
    blob->ImportFromExtenalData(fdata.data(), n_det*3*128*128);

    net->Forward();

    PrintBlob(net->blob("prob"), 100);
  }



  return 0;
}
