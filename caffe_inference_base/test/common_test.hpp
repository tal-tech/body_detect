#ifndef __FACETHINK_TEST_COMMON_HPP__
#define __FACETHINK_TEST_COMMON_HPP__

#include "caffe/caffe.hpp"

#include <fstream>
#include <iomanip>

template <typename Dtype>
void PrintBlob(const std::shared_ptr<facethink::Blob<Dtype> >& blob, int num){
  const Dtype* data = blob->cpu_data();
  int n = std::min(blob->count(), num);

  for (int i=0; i<n; ++i){
    std::cout.setf(std::ios::scientific, std::ios::floatfield);
    std::cout<<std::setprecision(8)<<data[i]<<"\t";

    if ((i+1) % 6 == 0)
      std::cout<<std::endl;
  }
  std::cout<<std::endl;
  std::cout<<blob->shape_string()<<std::endl;

  std::cout<<std::endl<<"**************************************"<<std::endl;
}



#endif
