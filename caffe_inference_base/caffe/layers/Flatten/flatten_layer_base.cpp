#include "caffe/layers/Flatten/flatten_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  void BaseFlattenLayer<Dtype>::ForwardShape() {
    this->CheckBlobs();

    const int num_axes   = this->inputs_[0]->num_axes();
    const int start_axis = this->inputs_[0]->CanonicalAxisIndex(start_axis_);
    const int end_axis   = this->inputs_[0]->CanonicalAxisIndex(end_axis_);

    const std::vector<int> input_shape = this->inputs_[0]->shape();
    std::vector<int> output_shape;

    for (int i = 0; i < start_axis; ++i){
      output_shape.push_back(input_shape[i]);
    }
    
    int flattened_dim = 1;
    for (int i = start_axis; i<end_axis+1; ++i){
      flattened_dim *= input_shape[i];
    }
    output_shape.push_back(flattened_dim);
    
    for (int i = end_axis+1; i < num_axes; ++i){
      output_shape.push_back(input_shape[i]);
    }

    this->outputs_[0]->Reshape(output_shape);
    this->outputs_[0]->ShareData(*this->inputs_[0]);
  }

  INSTANTIATE_CLASS(BaseFlattenLayer);
} //namespace facethink
