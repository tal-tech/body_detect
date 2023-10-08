#include "caffe/layers/Dropout/dropout_layer_cuda.hpp"
#include "caffe/util/math_func.hpp"

namespace facethink {

  template <typename Dtype>
  void CUDADropoutLayer<Dtype>::ForwardComputation() {
    const Dtype* input_data = this->inputs_[0]->gpu_data();
    Dtype* output_data = this->outputs_[0]->mutable_gpu_data();
    const int count = this->inputs_[0]->count();

    caffe_gpu_copy(count, input_data, output_data);
  }
  
  INSTANTIATE_CLASS(CUDADropoutLayer);
}
