#include "caffe/layers/Eltwise/eltwise_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  void BaseEltwiseLayer<Dtype>::ForwardShape() {
    for (size_t i = 1; i < this->inputs_.size(); ++i) {
      if (this->inputs_[0]->shape() != this->inputs_[i]->shape()){
	BOOST_LOG_TRIVIAL(error)<<"EltwiseLayer: shape mismatch "
				<< "input[0]: " << this->inputs_[0]->shape_string()
				<< ", input[" << i << "]: " << this->inputs_[i]->shape_string();
      } 
    }

    if (!coeffs_.empty() && coeffs_.size() != this->inputs_.size()) {
      BOOST_LOG_TRIVIAL(error)<<"EltwiseLayer: must takes one coefficient per bottom blob.";
    }

    if (op_ == EltwiseOp::PROD && coeffs_.size()) {
      BOOST_LOG_TRIVIAL(error)<<"EltwiseLayer: only takes coefficients for summation.";
    }
  
    this->outputs_[0]->Reshape(this->inputs_[0]->shape());
    // If max operation, we will initialize the vector index part.
    if (op_ == EltwiseOp::MAX && this->outputs_.size() == 1) {
      max_idx_.Reshape(this->inputs_[0]->shape());
    }
  }

  INSTANTIATE_CLASS(BaseEltwiseLayer);
} // namespace facethink 
