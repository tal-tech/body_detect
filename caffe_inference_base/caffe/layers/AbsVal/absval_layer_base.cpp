#include "caffe/layers/AbsVal/absval_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  void BaseAbsValLayer<Dtype>::ForwardShape() {
    this->CheckBlobs();

    const std::vector<int> input_shape = this->inputs_[0]->shape();
    this->outputs_[0]->Reshape(input_shape);
  }

  INSTANTIATE_CLASS(BaseAbsValLayer);
}
