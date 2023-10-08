#include "caffe/core/blob.hpp"
#include <vector>
#include <iostream>
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>

int main() {
  boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::info);
  {
    std::vector<int> shape {1, 3, 4000};
    facethink::Blob<float> blob(shape);
    BOOST_LOG_TRIVIAL(info)<<"Blob: "<<blob.shape_string()<<std::endl;
  
    // std::vector<float> input_data;
    // for (size_t i = 0; i < 12000; ++i){
    //   input_data.push_back(i);
    // }
    // blob.ImportFromExtenalData(input_data.data(), 12000);

#ifndef CPU_ONLY
    const float* gpu_data = blob.gpu_data();
#endif
    
    const float* cpu_data = blob.cpu_data();

    for (int i=0; i<1000; i++){
      std::cout<<cpu_data[i]<<"   ";
    }
    std::cout<<std::endl;
  
  }
  getchar();
  return 0;
}
