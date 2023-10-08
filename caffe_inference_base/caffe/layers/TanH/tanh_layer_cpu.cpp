#include "caffe/layers/TanH/tanh_layer_cpu.hpp"

namespace facethink {

  template <typename Dtype>
  void TanHLayer<Dtype>::ForwardShape() {
    this->CheckBlobs();
    
    const std::vector<int> input_shape = this->inputs_[0]->shape();
    std::vector<int> magic_input_shape = {1, 1, 1, 1};

    for (size_t i=0; i<input_shape.size(); ++i){
      magic_input_shape[i] = input_shape[i];
    }

    this->outputs_[0]->Reshape(input_shape);
  }

  template <typename Dtype>
  void TanHLayer<Dtype>::ForwardComputation() {
    const Dtype* input_data = this->inputs_[0]->cpu_data();
    Dtype* output_data = this->outputs_[0]->mutable_cpu_data();
    const int count = this->inputs_[0]->count();
    for (int i = 0; i < count; ++i) {
      output_data[i] = tanh(input_data[i]);
    }
  }

  INSTANTIATE_CLASS(TanHLayer);
} // namespace facethink
