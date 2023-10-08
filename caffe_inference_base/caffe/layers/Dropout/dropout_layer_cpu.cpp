#include "caffe/layers/Dropout/dropout_layer_cpu.hpp"
#include "caffe/util/math_func.hpp"

namespace facethink {

  template <typename Dtype>
  void DropoutLayer<Dtype>::ForwardComputation() {
    const Dtype* input_data = this->inputs_[0]->cpu_data();
    Dtype* output_data = this->outputs_[0]->mutable_cpu_data();
    const int count = this->inputs_[0]->count();

    caffe_cpu_copy(count, input_data, output_data);
  }
  
  INSTANTIATE_CLASS(DropoutLayer);
}
