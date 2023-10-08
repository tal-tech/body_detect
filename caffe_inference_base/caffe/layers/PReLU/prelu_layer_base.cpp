#include "caffe/layers/PReLU/prelu_layer_base.hpp"

namespace facethink {
  
  template <typename Dtype>
  void BasePReluLayer<Dtype>::ForwardShape() {
    this->CheckBlobs();
	this->CheckWeights();

    const std::vector<int>& input_shape = this->inputs_[0]->shape();
    this->outputs_[0]->Reshape(input_shape);

  }

  
  INSTANTIATE_CLASS(BasePReluLayer);
} // namespace facethink
