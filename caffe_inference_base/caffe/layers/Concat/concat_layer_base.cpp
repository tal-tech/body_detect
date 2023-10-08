#include "caffe/layers/Concat/concat_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  void BaseConcatLayer<Dtype>::ForwardShape() {
    this->CheckBlobs();

    concat_axis_canonical_ = this->inputs_[0]->CanonicalAxisIndex(concat_axis_);
    
    const int num_axes = this->inputs_[0]->num_axes();
    // Initialize with the first blob.
    std::vector<int> base_shape = this->inputs_[0]->shape();

    num_concats_ = this->inputs_[0]->count(0, concat_axis_canonical_);
    concat_input_size_ = this->inputs_[0]->count(concat_axis_canonical_ + 1);
    
    int input_count_sum = this->inputs_[0]->count();

    for (int i = 1; i < this->inputs_.size(); ++i) {
      if (this->inputs_[i]->num_axes() != num_axes){
	BOOST_LOG_TRIVIAL(error)<<"ConcatLayer: All inputs must have the same #axes.";
      }
      for (int j = 0; j < num_axes; ++j) {
	if (j == concat_axis_canonical_) { continue; }
	if (base_shape[j] !=  this->inputs_[i]->shape(j)){
	  BOOST_LOG_TRIVIAL(error)<<"ConcatLayer: All inputs must have the same shape, except at concat_axis.";
	}
      }
      input_count_sum += this->inputs_[i]->count();
      base_shape[concat_axis_canonical_] += this->inputs_[i]->shape(concat_axis_canonical_);
    }

    this->outputs_[0]->Reshape(base_shape);
    
    if (input_count_sum != this->outputs_[0]->count()){
      BOOST_LOG_TRIVIAL(error)<<"ConcatLayer: Error shape dismatch";
    }
    
    if (this->inputs_.size() == 1) {
      this->outputs_[0]->ShareData(*this->inputs_[0]);;
    }

  }

  INSTANTIATE_CLASS(BaseConcatLayer);
} // namespace facethink
