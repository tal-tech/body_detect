#include "caffe/layers/ReLU/relu_layer_cpu.hpp"

namespace facethink {
  
  template <typename Dtype>
  void ReluLayer<Dtype>::ForwardShape() {
    this->CheckBlobs();

    const std::vector<int>& input_shape = this->inputs_[0]->shape();
    this->outputs_[0]->Reshape(input_shape);

  }

  template <typename Dtype>
  void ReluLayer<Dtype>::ForwardComputation() {
    const Dtype* input_data = this->inputs_[0]->cpu_data();
    Dtype* output_data = this->outputs_[0]->mutable_cpu_data();
    const int count = this->inputs_[0]->count();
    
    for (int i = 0; i < count; ++i) {
      output_data[i] = std::max(input_data[i], Dtype(0))
	+ this->negative_slope_ * std::min(input_data[i], Dtype(0));
    }
  }
  
  INSTANTIATE_CLASS(ReluLayer);
} // namespace facethink
