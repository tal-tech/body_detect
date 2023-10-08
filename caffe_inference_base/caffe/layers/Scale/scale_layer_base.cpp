#include "caffe/layers/Scale/scale_layer_base.hpp"
#include "caffe/util/math_func.hpp"

namespace facethink {

  template <typename Dtype>
  void BaseScaleLayer<Dtype>::ForwardShape() {
    this->CheckBlobs();
    this->CheckWeights();
    
    std::shared_ptr<Blob<Dtype> > scale = (this->inputs_.size() > 1) ?
      this->inputs_[1] : this->weights_[0];
    
    // Always set axis_ == 0 in special case where scale is a scalar
    // (num_axes == 0). Mathematically equivalent for any choice of axis_, so the
    // actual setting can be safely ignored; and computation is most efficient
    // with axis_ == 0 and (therefore) outer_dim_ == 1. (Setting axis_ to
    // bottom[0]->num_axes() - 1, giving inner_dim_ == 1, would be equally
    // performant.)
    axis_valid_ = (scale->num_axes() == 0) ?
      0 : this->inputs_[0]->CanonicalAxisIndex(axis_);
    
    if (this->inputs_[0]->num_axes() < axis_valid_ + scale->num_axes()) {
      BOOST_LOG_TRIVIAL(error)<<"scale blob's shape extends past bottom[0]'s shape when applied "
			      <<"starting with bottom[0] axis = " << axis_valid_;
    }

    for (int i = 0; i < scale->num_axes(); ++i) {
      if (this->inputs_[0]->shape(axis_valid_ + i) != scale->shape(i)) {
	 BOOST_LOG_TRIVIAL(error)<<"dimension mismatch between bottom[0]->shape(" << axis_valid_ + i
				 << ") and scale->shape(" << i << ")";
      }
    }
    
    outer_dim_ = this->inputs_[0]->count(0, axis_valid_);
    scale_dim_ = scale->count();
    inner_dim_ = this->inputs_[0]->count(axis_valid_ + scale->num_axes());

    this->outputs_[0]->Reshape(this->inputs_[0]->shape());
  }
  
  INSTANTIATE_CLASS(BaseScaleLayer);
} // namespace facethink
